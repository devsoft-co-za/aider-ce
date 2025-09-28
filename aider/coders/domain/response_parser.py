#!/usr/bin/env python

import json
import os
import re
from collections import defaultdict
from json.decoder import JSONDecodeError
from pathlib import Path


class ResponseParser:
    """Manages parsing and processing of LLM responses for the Coder class."""

    def __init__(self, coder):
        self.coder = coder

    def parse_partial_args(self):
        """Parse arguments from partial function call response."""
        data = self.coder.partial_response_function_call.get("arguments")
        if not data:
            return

        try:
            return json.loads(data)
        except JSONDecodeError:
            pass

        try:
            return json.loads(data + "]}")
        except JSONDecodeError:
            pass

        try:
            return json.loads(data + "}]}")
        except JSONDecodeError:
            pass

        try:
            return json.loads(data + '"}]}')
        except JSONDecodeError:
            pass

    def _find_occurrences(self, content, pattern, near_context=None):
        """Find all occurrences of pattern, optionally filtered by near_context."""
        occurrences = []
        start = 0
        while True:
            index = content.find(pattern, start)
            if index == -1:
                break

            if near_context:
                # Check if near_context is within a window around the match
                window_start = max(0, index - 200)
                window_end = min(len(content), index + len(pattern) + 200)
                window = content[window_start:window_end]
                if near_context in window:
                    occurrences.append(index)
            else:
                occurrences.append(index)

            start = index + 1  # Move past this occurrence's start
        return occurrences

    def get_file_mentions(self, content, ignore_current=False):
        """Extract file mentions from content."""
        words = set(word for word in content.split())

        # drop sentence punctuation from the end
        words = set(word.rstrip(",.!;:?") for word in words)

        # strip away all kinds of quotes
        quotes = "\"'`*_"
        words = set(word.strip(quotes) for word in words)

        if ignore_current:
            addable_rel_fnames = self.coder.get_all_relative_files()
            existing_basenames = {}
        else:
            addable_rel_fnames = self.coder.get_addable_relative_files()

            # Get basenames of files already in chat or read-only
            existing_basenames = {
                os.path.basename(f) for f in self.coder.get_inchat_relative_files()
            } | {
                os.path.basename(self.coder.get_rel_fname(f))
                for f in self.coder.abs_read_only_fnames | self.coder.abs_read_only_stubs_fnames
            }

        mentioned_rel_fnames = set()
        fname_to_rel_fnames = {}
        for rel_fname in addable_rel_fnames:
            normalized_rel_fname = rel_fname.replace("\\", "/")
            normalized_words = set(word.replace("\\", "/") for word in words)
            if normalized_rel_fname in normalized_words:
                mentioned_rel_fnames.add(rel_fname)

            fname = os.path.basename(rel_fname)

            # Don't add basenames that could be plain words like "run" or "make"
            if "/" in fname or "\\" in fname or "." in fname or "_" in fname or "-" in fname:
                if fname not in fname_to_rel_fnames:
                    fname_to_rel_fnames[fname] = []
                fname_to_rel_fnames[fname].append(rel_fname)

        for fname, rel_fnames in fname_to_rel_fnames.items():
            # If the basename is already in chat, don't add based on a basename mention
            if fname in existing_basenames:
                continue
            # If the basename mention is unique among addable files and present in the text
            if len(rel_fnames) == 1 and fname in words:
                mentioned_rel_fnames.add(rel_fnames[0])

        return mentioned_rel_fnames

    def get_ident_mentions(self, text):
        """Extract identifier mentions from text."""
        # Split the string on any character that is not alphanumeric
        # \W+ matches one or more non-word characters (equivalent to [^a-zA-Z0-9_]+)
        words = set(re.split(r"\W+", text))
        return words

    def get_ident_filename_matches(self, idents):
        """Match identifiers to filenames."""
        all_fnames = defaultdict(set)
        for fname in self.coder.get_all_relative_files():
            # Skip empty paths or just '.'
            if not fname or fname == ".":
                continue

            try:
                # Handle dotfiles properly
                path = Path(fname)
                base = path.stem.lower()  # Use stem instead of with_suffix("").name
                if len(base) >= 5:
                    all_fnames[base].add(fname)
            except ValueError:
                # Skip paths that can't be processed
                continue

        matches = set()
        for ident in idents:
            if len(ident) < 5:
                continue
            matches.update(all_fnames[ident.lower()])

        return matches
