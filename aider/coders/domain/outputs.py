#!/usr/bin/env python

import math


class OutputManager:
    """Manages output formatting and display for the Coder class."""

    def __init__(self, coder):
        self.coder = coder

    def show_announcements(self):
        """Display startup announcements."""
        bold = True
        for line in self.coder.get_announcements():
            self.coder.io.tool_output(line, bold=bold)
            bold = False

    def show_pretty(self):
        """Check if pretty output should be shown."""
        if not self.coder.pretty:
            return False

        # only show pretty output if fences are the normal triple-backtick
        if self.coder.fence[0][0] != "`":
            return False

        return True

    def show_exhausted_error(self):
        """Display error when context window is exhausted."""
        output_tokens = 0
        if self.coder.partial_response_content:
            output_tokens = self.coder.main_model.token_count(self.coder.partial_response_content)
        max_output_tokens = self.coder.main_model.info.get("max_output_tokens") or 0

        input_tokens = self.coder.main_model.token_count(
            self.coder.format_messages().all_messages()
        )
        max_input_tokens = self.coder.main_model.info.get("max_input_tokens") or 0

        total_tokens = input_tokens + output_tokens

        fudge = 0.7

        out_err = ""
        if output_tokens >= max_output_tokens * fudge:
            out_err = " -- possibly exceeded output limit!"

        inp_err = ""
        if input_tokens >= max_input_tokens * fudge:
            inp_err = " -- possibly exhausted context window!"

        tot_err = ""
        if total_tokens >= max_input_tokens * fudge:
            tot_err = " -- possibly exhausted context window!"

        res = ["", ""]
        res.append(f"Model {self.coder.main_model.name} has hit a token limit!")
        res.append("Token counts below are approximate.")
        res.append("")
        res.append(f"Input tokens: ~{input_tokens:,} of {max_input_tokens:,}{inp_err}")
        res.append(f"Output tokens: ~{output_tokens:,} of {max_output_tokens:,}{out_err}")
        res.append(f"Total tokens: ~{total_tokens:,} of {max_input_tokens:,}{tot_err}")

        if output_tokens >= max_output_tokens:
            res.append("")
            res.append("To reduce output tokens:")
            res.append("- Ask for smaller changes in each request.")
            res.append("- Break your code into smaller source files.")
            if "diff" not in self.coder.main_model.edit_format:
                res.append("- Use a stronger model that can return diffs.")

        if input_tokens >= max_input_tokens or total_tokens >= max_input_tokens:
            res.append("")
            res.append("To reduce input tokens:")
            res.append("- Use /tokens to see token usage.")
            res.append("- Use /drop to remove unneeded files from the chat session.")
            res.append("- Use /clear to clear the chat history.")
            res.append("- Break your code into smaller source files.")

        res = "".join([line + "\n" for line in res])
        self.coder.io.tool_error(res)
        self.coder.io.offer_url(self.coder.urls.token_limits)

    def show_auto_commit_outcome(self, res):
        """Display the outcome of an auto-commit."""
        commit_hash, commit_message = res
        self.coder.last_aider_commit_hash = commit_hash
        self.coder.aider_commit_hashes.add(commit_hash)
        self.coder.last_aider_commit_message = commit_message
        if self.coder.show_diffs:
            self.coder.commands.cmd_diff()

    def show_undo_hint(self):
        """Display hint about undo functionality."""
        if not self.coder.commit_before_message:
            return
        if self.coder.commit_before_message[-1] != self.coder.repo.get_head_commit_sha():
            self.coder.io.tool_output("You can use /undo to undo and discard each aider commit.")

    def show_usage_report(self):
        """Display token usage and cost report."""
        if not self.coder.usage_report:
            return

        self.coder.total_tokens_sent += self.coder.message_tokens_sent
        self.coder.total_tokens_received += self.coder.message_tokens_received

        self.coder.io.tool_output(self.coder.usage_report)

        prompt_tokens = self.coder.message_tokens_sent
        completion_tokens = self.coder.message_tokens_received
        self.coder.event(
            "message_send",
            main_model=self.coder.main_model,
            edit_format=self.coder.edit_format,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=self.coder.message_cost,
            total_cost=self.coder.total_cost,
        )

        self.coder.message_cost = 0.0
        self.coder.message_tokens_sent = 0
        self.coder.message_tokens_received = 0

    def format_cost(self, value):
        """Format cost value for display."""
        if value == 0:
            return "0.00"
        magnitude = abs(value)
        if magnitude >= 0.01:
            return f"{value:.2f}"
        else:
            return f"{value:.{max(2, 2 - int(math.log10(magnitude)))}f}"

    def get_platform_info(self):
        """Get platform and environment information."""
        platform_text = ""
        try:
            platform_text = f"- Platform: {self.coder.platform.platform()}\n"
        except KeyError:
            # Skip platform info if it can't be retrieved
            platform_text = "- Platform information unavailable\n"

        shell_var = "COMSPEC" if self.coder.os.name == "nt" else "SHELL"
        shell_val = self.coder.os.getenv(shell_var)
        platform_text += f"- Shell: {shell_var}={shell_val}\n"

        user_lang = self.coder.get_user_language()
        if user_lang:
            platform_text += f"- Language: {user_lang}\n"

        dt = self.coder.datetime.now().astimezone().strftime("%Y-%m-%d")
        platform_text += f"- Current date: {dt}\n"

        if self.coder.repo:
            platform_text += "- The user is operating inside a git repository\n"

        if self.coder.lint_cmds:
            if self.coder.auto_lint:
                platform_text += (
                    "- The user's pre-commit runs these lint commands, don't suggest running"
                    " them:\n"
                )
            else:
                platform_text += "- The user prefers these lint commands:\n"
            for lang, cmd in self.coder.lint_cmds.items():
                if lang is None:
                    platform_text += f"  - {cmd}\n"
                else:
                    platform_text += f"  - {lang}: {cmd}\n"

        if self.coder.test_cmd:
            if self.coder.auto_test:
                platform_text += (
                    "- The user's pre-commit runs this test command, don't suggest running them: "
                )
            else:
                platform_text += "- The user prefers this test command: "
            platform_text += self.coder.test_cmd + "\n"

        return platform_text
