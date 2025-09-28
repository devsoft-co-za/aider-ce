#!/usr/bin/env python

import os
from pathlib import Path

from aider import utils


class FileManager:
    """Manages file operations and state for the Coder class."""

    def __init__(self, coder):
        self.coder = coder
        self.abs_fnames = set()
        self.abs_read_only_fnames = set()
        self.abs_read_only_stubs_fnames = set()
        self.abs_root_path_cache = {}
        self.warning_given = False

    def add_rel_fname(self, rel_fname):
        """Add a relative filename to the chat."""
        self.abs_fnames.add(self.abs_root_path(rel_fname))
        self.check_added_files()

    def drop_rel_fname(self, fname):
        """Remove a relative filename from the chat."""
        abs_fname = self.abs_root_path(fname)
        if abs_fname in self.abs_fnames:
            self.abs_fnames.remove(abs_fname)
            return True

    def abs_root_path(self, path):
        """Convert a relative path to an absolute path relative to the root."""
        key = path
        if key in self.abs_root_path_cache:
            return self.abs_root_path_cache[key]

        res = Path(self.coder.root) / path
        res = utils.safe_abs_path(res)
        self.abs_root_path_cache[key] = res
        return res

    def get_abs_fnames_content(self):
        """Generator that yields (fname, content) for all files in the chat."""
        for fname in list(self.abs_fnames):
            content = self.coder.io.read_text(fname)

            if content is None:
                relative_fname = self.get_rel_fname(fname)
                self.coder.io.tool_warning(f"Dropping {relative_fname} from the chat.")
                self.abs_fnames.remove(fname)
            else:
                yield fname, content

    def get_files_content(self, fnames=None):
        """Get the content of all files in the chat."""
        if not fnames:
            fnames = self.abs_fnames

        prompt = ""
        for fname, content in self.get_abs_fnames_content():
            if not utils.is_image_file(fname):
                relative_fname = self.get_rel_fname(fname)
                prompt += "\n"
                prompt += relative_fname
                prompt += f"\n{self.coder.fence[0]}\n"

                # Apply context management if enabled for large files
                if self.coder.context_management_enabled:
                    # Calculate tokens for this file
                    file_tokens = self.coder.main_model.token_count(content)

                    if file_tokens > self.coder.large_file_token_threshold:
                        # Truncate the file content
                        lines = content.splitlines()

                        # Keep the first and last parts of the file with a marker in between
                        keep_lines = (
                            self.coder.large_file_token_threshold // 40
                        )  # Rough estimate of tokens per line
                        first_chunk = lines[: keep_lines // 2]
                        last_chunk = lines[-(keep_lines // 2) :]

                        truncated_content = "\n".join(first_chunk)
                        truncated_content += (
                            f"\n\n... [File truncated due to size ({file_tokens} tokens). Use"
                            " /context-management to toggle truncation off] ...\n\n"
                        )
                        truncated_content += "\n".join(last_chunk)

                        # Add message about truncation
                        self.coder.io.tool_output(
                            f"⚠️ '{relative_fname}' is very large ({file_tokens} tokens). "
                            "Use /context-management to toggle truncation off if needed."
                        )

                        prompt += truncated_content
                    else:
                        prompt += content
                else:
                    prompt += content

                prompt += f"{self.coder.fence[1]}\n"

        return prompt

    def get_read_only_files_content(self):
        """Get the content of all read-only files."""
        prompt = ""
        # Handle regular read-only files
        for fname in self.abs_read_only_fnames:
            content = self.coder.io.read_text(fname)
            if content is not None and not utils.is_image_file(fname):
                relative_fname = self.get_rel_fname(fname)
                prompt += "\n"
                prompt += relative_fname
                prompt += f"\n{self.coder.fence[0]}\n"

                # Apply context management if enabled for large files (same as get_files_content)
                if self.coder.context_management_enabled:
                    # Calculate tokens for this file
                    file_tokens = self.coder.main_model.token_count(content)

                    if file_tokens > self.coder.large_file_token_threshold:
                        # Truncate the file content
                        lines = content.splitlines()

                        # Keep the first and last parts of the file with a marker in between
                        keep_lines = (
                            self.coder.large_file_token_threshold // 40
                        )  # Rough estimate of tokens per line
                        first_chunk = lines[: keep_lines // 2]
                        last_chunk = lines[-(keep_lines // 2) :]

                        truncated_content = "\n".join(first_chunk)
                        truncated_content += (
                            f"\n\n... [File truncated due to size ({file_tokens} tokens). Use"
                            " /context-management to toggle truncation off] ...\n\n"
                        )
                        truncated_content += "\n".join(last_chunk)

                        # Add message about truncation
                        self.coder.io.tool_output(
                            f"⚠️ '{relative_fname}' is very large ({file_tokens} tokens). "
                            "Use /context-management to toggle truncation off if needed."
                        )

                        prompt += truncated_content
                    else:
                        prompt += content
                else:
                    prompt += content

                prompt += f"{self.coder.fence[1]}\n"

        # Handle stub files
        for fname in self.abs_read_only_stubs_fnames:
            if not utils.is_image_file(fname):
                relative_fname = self.get_rel_fname(fname)
                prompt += "\n"
                prompt += f"{relative_fname} (stub)"
                prompt += f"\n{self.coder.fence[0]}\n"
                stub = self.get_file_stub(fname)
                prompt += stub
                prompt += f"{self.coder.fence[1]}\n"
        return prompt

    def get_rel_fname(self, fname):
        """Get the relative filename from the root."""
        try:
            return os.path.relpath(fname, self.coder.root)
        except ValueError:
            return fname

    def get_inchat_relative_files(self):
        """Get all relative filenames currently in the chat."""
        files = [self.get_rel_fname(fname) for fname in self.abs_fnames]
        return sorted(set(files))

    def is_file_safe(self, fname):
        """Check if a file is safe to access."""
        try:
            return Path(self.abs_root_path(fname)).is_file()
        except OSError:
            return

    def get_all_relative_files(self):
        """Get all relative files in the repository."""
        if self.coder.repo:
            files = self.coder.repo.get_tracked_files()
        else:
            files = self.get_inchat_relative_files()

        # This is quite slow in large repos
        # files = [fname for fname in files if self.is_file_safe(fname)]

        return sorted(set(files))

    def get_all_abs_files(self):
        """Get all absolute files in the repository."""
        files = self.get_all_relative_files()
        files = [self.abs_root_path(path) for path in files]
        return files

    def get_addable_relative_files(self):
        """Get all relative files that can be added to the chat."""
        all_files = set(self.get_all_relative_files())
        inchat_files = set(self.get_inchat_relative_files())
        read_only_files = set(self.get_rel_fname(fname) for fname in self.abs_read_only_fnames)
        stub_files = set(self.get_rel_fname(fname) for fname in self.abs_read_only_stubs_fnames)
        return all_files - inchat_files - read_only_files - stub_files

    def get_file_stub(self, fname):
        """Get a stub representation of a file."""
        return self.coder.repo_map.get_file_stub(fname, self.coder.io)

    def check_added_files(self):
        """Check if too many files have been added and warn the user."""
        if self.warning_given:
            return

        warn_number_of_files = 4
        warn_number_of_tokens = 20 * 1024

        num_files = len(self.abs_fnames)
        if num_files < warn_number_of_files:
            return

        tokens = 0
        for fname in self.abs_fnames:
            if utils.is_image_file(fname):
                continue
            content = self.coder.io.read_text(fname)
            tokens += self.coder.main_model.token_count(content)

        if tokens < warn_number_of_tokens:
            return

        self.coder.io.tool_warning(
            "Warning: it's best to only add files that need changes to the chat."
        )
        self.coder.io.tool_warning(self.coder.urls.edit_errors)
        self.warning_given = True
