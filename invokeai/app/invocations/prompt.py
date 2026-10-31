from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation
from invokeai.app.invocations.fields import InputField, UIComponent
from invokeai.app.invocations.primitives import StringCollectionOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.dynamicprompts import expand_dynamic_prompt
from invokeai.app.util.path_safety import resolve_contained_path

if TYPE_CHECKING:
    from invokeai.app.services.invocation_services import InvocationServices

_PROMPT_FILES_DIR_HINT = "the prompt files directory (`prompt_files_dir`, default `<root>/prompt_files`)"


@invocation(
    "dynamic_prompt",
    title="Dynamic Prompt",
    tags=["prompt", "collection"],
    category="prompt",
    # 1.1.0: wildcards now resolve against the queueing user's records, and an
    # unresolvable one fails the invocation instead of silently repeating a prompt.
    version="1.1.0",
    use_cache=False,
)
class DynamicPromptInvocation(BaseInvocation):
    """Parses a prompt using adieyal/dynamicprompts' random or combinatorial generator"""

    prompt: str = InputField(
        description="The prompt to parse with dynamicprompts",
        ui_component=UIComponent.Textarea,
    )
    max_prompts: int = InputField(default=1, description="The number of prompts to generate")
    combinatorial: bool = InputField(default=False, description="Whether to use the combinatorial generator")

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        expanded = expand_dynamic_prompt(
            self.prompt,
            max_prompts=self.max_prompts,
            combinatorial=self.combinatorial,
            wildcard_manager=context.wildcards.get_manager(),
        )

        # Unlike the HTTP route, a graph cannot usefully carry on with an unexpanded prompt.
        if expanded.error is not None:
            raise ValueError(expanded.error)

        return StringCollectionOutput(collection=expanded.prompts)


@invocation(
    "prompt_from_file",
    title="Prompts from File",
    tags=["prompt", "file"],
    category="prompt",
    # 2.0.0: `file_path` is now relative to the configured prompt files directory and cannot
    # address anything outside it. Absolute paths, which earlier versions accepted, are refused.
    # Major, so the editor does not carry an old `file_path` over as if it were still valid.
    version="2.0.0",
    use_cache=False,
)
class PromptsFromFileInvocation(BaseInvocation):
    """Loads prompts from a text file.

    The file is read from the InvokeAI prompt files directory (`prompt_files_dir`, by default
    `<root>/prompt_files`). `file_path` is relative to that directory - absolute paths and path
    traversal are rejected - so the node can only read prompt lists an operator has deliberately
    placed there.

    In multi-user mode the node is restricted to administrators: the directory is shared by every
    account, so letting one account read from it would hand it another account's prompt lists.
    Non-admin users should use wildcards, which are stored per account.
    """

    file_path: str = InputField(description="Path to prompt text file, relative to the InvokeAI prompt files directory")
    pre_prompt: Optional[str] = InputField(
        default=None,
        description="String to prepend to each prompt",
        ui_component=UIComponent.Textarea,
    )
    post_prompt: Optional[str] = InputField(
        default=None,
        description="String to append to each prompt",
        ui_component=UIComponent.Textarea,
    )
    start_line: int = InputField(default=1, ge=1, description="Line in the file to start start from")
    max_prompts: int = InputField(default=1, ge=0, description="Max lines to read from file (0=all)")

    def _resolve_file_path(self, context: InvocationContext) -> Path:
        """Resolves `file_path` inside the prompt files directory, after checking the caller may read it.

        Deliberately *not* a pydantic validator. A validator runs while the enqueue request body is
        being parsed, which means every failure - including "that file does not exist" - is answered
        as a 422 before anything is queued, turning the node into a filesystem probe that leaves no
        queue item behind. Resolving here instead makes a bad path fail as an error on the caller's
        own queue item, the same as any other invocation failure.
        """
        config = context._services.configuration

        if config.multiuser:
            # The prompt files directory is shared by every account, so reading from it is an admin
            # capability. `call_saved_workflow` gates on the queueing user the same way.
            queue_user_id = context._data.queue_item.user_id
            user = context._services.users.get(queue_user_id)
            if not (user and user.is_admin):
                raise ValueError(
                    "The 'Prompts from File' node is restricted to administrators in multi-user mode. "
                    "Use wildcards instead - they are stored per account."
                )

        # Errors name the setting, not the resolved path: the absolute path is server-side detail, and
        # echoing it back would describe the host's layout to whoever enqueued the graph.
        try:
            resolved = resolve_contained_path(config.prompt_files_path, self.file_path)
        except ValueError as e:
            raise ValueError(
                f"Invalid prompt file path: {e} 'file_path' must be relative to {_PROMPT_FILES_DIR_HINT}."
            ) from e

        if not resolved.is_file():
            raise ValueError(
                f"No prompt file '{self.file_path}' in {_PROMPT_FILES_DIR_HINT}. "
                "Place prompt list files there to use them with this node."
            )

        return resolved

    def promptsFromFile(
        self,
        file_path: Union[str, Path],
        pre_prompt: Union[str, None],
        post_prompt: Union[str, None],
        start_line: int,
        max_prompts: int,
    ):
        prompts = []
        start_line -= 1
        end_line = start_line + max_prompts
        if max_prompts <= 0:
            end_line = np.iinfo(np.int32).max
        with open(file_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= start_line and i < end_line:
                    prompts.append((pre_prompt or "") + line.strip() + (post_prompt or ""))
                if i >= end_line:
                    break
        return prompts

    def invoke_internal(self, context: InvocationContext, services: "InvocationServices") -> BaseInvocationOutput:
        # Never use the invocation cache: the admin gate lives in invoke() and cache keys carry no user
        # identity, so a hit would hand an admin's read to anyone (and the file can change under the key).
        self._validate_invoke_fields()
        return self.invoke(context)

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        prompts = self.promptsFromFile(
            self._resolve_file_path(context),
            self.pre_prompt,
            self.post_prompt,
            self.start_line,
            self.max_prompts,
        )
        return StringCollectionOutput(collection=prompts)
