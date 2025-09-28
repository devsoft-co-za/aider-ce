#!/usr/bin/env python

import asyncio
import os
import threading
import time

from aider import utils
from aider.llm import litellm


class RuntimeManager:
    """Manages runtime operations and state for the Coder class."""

    def __init__(self, coder):
        self.coder = coder
        self.num_exhausted_context_windows = 0
        self.num_malformed_responses = 0
        self.last_keyboard_interrupt = None
        self.num_reflections = 0
        self.max_reflections = 3
        self.num_tool_calls = 0
        self.max_tool_calls = 25
        self.run_one_completed = True
        self.compact_context_completed = True
        self.confirmation_in_progress = False
        self.tool_reflection = False

        # Context management settings
        self.context_management_enabled = False
        self.large_file_token_threshold = 25000

        # Cache warming
        self.add_cache_headers = False
        self.cache_warming_thread = None
        self.num_cache_warming_pings = 0
        self.ok_to_warm_cache = False

        # Summarization
        self.summarizer_thread = None
        self.summarized_done_messages = []
        self.summarizing_messages = None
        self.input_task = None

    def init_before_message(self):
        """Initialize state before processing a new message."""
        self.coder.aider_edited_files = set()
        self.coder.reflected_message = None
        self.num_reflections = 0
        self.coder.lint_outcome = None
        self.coder.test_outcome = None
        self.coder.shell_commands = []
        self.coder.message_cost = 0

        if self.coder.repo:
            self.coder.commit_before_message.append(self.coder.repo.get_head_commit_sha())

    def keyboard_interrupt(self):
        """Handle keyboard interrupt."""
        from rich.console import Console

        Console().show_cursor(True)
        self.coder.io.tool_warning("\n\n^C KeyboardInterrupt")
        self.last_keyboard_interrupt = time.time()

    def summarize_start(self):
        """Start chat history summarization in a background thread."""
        if not self.coder.summarizer.check_max_tokens(self.coder.done_messages):
            return

        self.summarize_end()

        if self.coder.verbose:
            self.coder.io.tool_output("Starting to summarize chat history.")

        self.summarizer_thread = threading.Thread(target=self.summarize_worker)
        self.summarizer_thread.start()

    def summarize_worker(self):
        """Worker thread for chat history summarization."""
        self.summarizing_messages = list(self.coder.done_messages)
        try:
            self.summarized_done_messages = asyncio.run(
                self.coder.summarizer.summarize(self.summarizing_messages)
            )
        except ValueError as err:
            self.coder.io.tool_warning(err.args[0])
            self.summarized_done_messages = self.summarizing_messages

        if self.coder.verbose:
            self.coder.io.tool_output("Finished summarizing chat history.")

    def summarize_end(self):
        """End chat history summarization and update messages."""
        if self.summarizer_thread is None:
            return

        self.summarizer_thread.join()
        self.summarizer_thread = None

        if self.summarizing_messages == self.coder.done_messages:
            self.coder.done_messages = self.summarized_done_messages
        self.summarizing_messages = None
        self.summarized_done_messages = []

    async def compact_context_if_needed(self):
        """Compact chat context if it exceeds token limits."""
        if not self.coder.enable_context_compaction:
            self.summarize_start()
            return

        if not self.coder.summarizer.check_max_tokens(
            self.coder.done_messages, max_tokens=self.coder.context_compaction_max_tokens
        ):
            return

        self.coder.io.tool_output("Compacting chat history to make room for new messages...")

        try:
            summary_text = await self.coder.summarizer.summarize_all_as_text(
                self.coder.done_messages,
                self.coder.gpt_prompts.compaction_prompt,
                self.coder.context_compaction_summary_tokens,
            )
            if not summary_text:
                raise ValueError("Summarization returned an empty result.")

            self.coder.done_messages = [
                {
                    "role": "user",
                    "content": summary_text,
                },
                {
                    "role": "assistant",
                    "content": (
                        "Ok, I will use this summary as the context for our conversation going"
                        " forward."
                    ),
                },
            ]
            self.coder.io.tool_output("...chat history compacted.")
        except Exception as e:
            self.coder.io.tool_warning(f"Context compaction failed: {e}")
            self.coder.io.tool_warning("Proceeding with full history for now.")
            self.summarize_start()
            return

    def warm_cache(self, chunks):
        """Warm the prompt cache if enabled."""
        if not self.add_cache_headers:
            return
        if not self.num_cache_warming_pings:
            return
        if not self.ok_to_warm_cache:
            return

        delay = 5 * 60 - 5
        delay = float(os.environ.get("AIDER_CACHE_KEEPALIVE_DELAY", delay))
        self.next_cache_warm = time.time() + delay
        self.warming_pings_left = self.num_cache_warming_pings
        self.cache_warming_chunks = chunks

        if self.cache_warming_thread:
            return

        def warm_cache_worker():
            while self.ok_to_warm_cache:
                time.sleep(1)
                if self.warming_pings_left <= 0:
                    continue
                now = time.time()
                if now < self.next_cache_warm:
                    continue

                self.warming_pings_left -= 1
                self.next_cache_warm = time.time() + delay

                kwargs = dict(self.coder.main_model.extra_params) or dict()
                kwargs["max_tokens"] = 1

                try:
                    completion = litellm.completion(
                        model=self.coder.main_model.name,
                        messages=self.cache_warming_chunks.cacheable_messages(),
                        stream=False,
                        **kwargs,
                    )
                except Exception as err:
                    self.coder.io.tool_warning(f"Cache warming error: {str(err)}")
                    continue

                cache_hit_tokens = getattr(
                    completion.usage, "prompt_cache_hit_tokens", 0
                ) or getattr(completion.usage, "cache_read_input_tokens", 0)

                if self.coder.verbose:
                    self.coder.io.tool_output(
                        f"Warmed {utils.format_tokens(cache_hit_tokens)} cached tokens."
                    )

        self.cache_warming_thread = threading.Timer(0, warm_cache_worker)
        self.cache_warming_thread.daemon = True
        self.cache_warming_thread.start()

        return chunks

    def _stop_waiting_spinner(self):
        """Stop and clear the waiting spinner if it is running."""
        self.coder.io.stop_spinner()
