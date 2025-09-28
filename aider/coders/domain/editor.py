#!/usr/bin/env python

from pathlib import Path

from aider.repo import ANY_GIT_ERROR


class EditorManager:
    """Manages file editing operations for the Coder class."""

    def __init__(self, coder):
        self.coder = coder

    async def allowed_to_edit(self, path):
        """Check if editing is allowed for a given path."""
        full_path = self.coder.abs_root_path(path)
        if self.coder.repo:
            need_to_add = not self.coder.repo.path_in_repo(path)
        else:
            need_to_add = False

        if full_path in self.coder.abs_fnames:
            self.check_for_dirty_commit(path)
            return True

        if self.coder.repo and self.coder.repo.git_ignored_file(path):
            self.coder.io.tool_warning(f"Skipping edits to {path} that matches gitignore spec.")
            return

        if not Path(full_path).exists():
            if not self.coder.io.confirm_ask("Create new file?", subject=path):
                self.coder.io.tool_output(f"Skipping edits to {path}")
                return

            if not self.coder.dry_run:
                if not self.coder.utils.touch_file(full_path):
                    self.coder.io.tool_error(f"Unable to create {path}, skipping edits.")
                    return

                # Seems unlikely that we needed to create the file, but it was
                # actually already part of the repo.
                # But let's only add if we need to, just to be safe.
                if need_to_add:
                    self.coder.repo.repo.git.add(full_path)

            self.coder.abs_fnames.add(full_path)
            self.coder.check_added_files()
            return True

        if not await self.coder.io.confirm_ask(
            "Allow edits to file that has not been added to the chat?",
            subject=path,
        ):
            self.coder.io.tool_output(f"Skipping edits to {path}")
            return

        if need_to_add:
            self.coder.repo.repo.git.add(full_path)

        self.coder.abs_fnames.add(full_path)
        self.coder.check_added_files()
        self.check_for_dirty_commit(path)

        return True

    def check_for_dirty_commit(self, path):
        """Check if a file needs to be committed before editing."""
        if not self.coder.repo:
            return
        if not self.coder.dirty_commits:
            return
        if not self.coder.repo.is_dirty(path):
            return

        # We need a committed copy of the file in order to /undo, so skip this
        # fullp = Path(self.coder.abs_root_path(path))
        # if not fullp.stat().st_size:
        #     return

        self.coder.io.tool_output(f"Committing {path} before applying edits.")
        self.coder.need_commit_before_edits.add(path)

    async def prepare_to_edit(self, edits):
        """Prepare files for editing."""
        res = []
        seen = dict()

        self.coder.need_commit_before_edits = set()

        for edit in edits:
            path = edit[0]
            if path is None:
                res.append(edit)
                continue
            if path == "python":
                pass  # dump(edits)  # Removed unused dump import
            if path in seen:
                allowed = seen[path]
            else:
                allowed = await self.allowed_to_edit(path)
                seen[path] = allowed

            if allowed:
                res.append(edit)

        self.dirty_commit()
        self.coder.need_commit_before_edits = set()

        return res

    def dirty_commit(self):
        """Commit dirty files before editing."""
        if not self.coder.need_commit_before_edits:
            return
        if not self.coder.dirty_commits:
            return
        if not self.coder.repo:
            return

        self.coder.repo.commit(fnames=self.coder.need_commit_before_edits, coder=self.coder)

        # files changed, move cur messages back behind the files messages
        # self.coder.move_back_cur_messages(self.coder.gpt_prompts.files_content_local_edits)
        return True

    def get_context_from_history(self, history):
        """Get context from chat history for commit messages."""
        context = ""
        if history:
            for msg in history:
                msg_content = msg.get("content") or ""
                context += "\n" + msg["role"].upper() + ": " + msg_content + "\n"

        return context

    def auto_commit(self, edited, context=None):
        """Automatically commit edited files."""
        if not self.coder.repo or not self.coder.auto_commits or self.coder.dry_run:
            return

        if not context:
            context = self.get_context_from_history(self.coder.cur_messages)

        try:
            res = self.coder.repo.commit(
                fnames=edited, context=context, aider_edits=True, coder=self.coder
            )
            if res:
                self.coder.show_auto_commit_outcome(res)
                commit_hash, commit_message = res
                return self.coder.gpt_prompts.files_content_gpt_edits.format(
                    hash=commit_hash,
                    message=commit_message,
                )

            return self.coder.gpt_prompts.files_content_gpt_no_edits
        except ANY_GIT_ERROR as err:
            self.coder.io.tool_error(f"Unable to commit: {str(err)}")
            return
