import type { GenerateLora, GenerateModelConfig } from '@workbench/generation/types';
import type { PromptHistoryItem } from '@workbench/types';
import type { ChangeEvent, KeyboardEvent } from 'react';

import { Field } from '@workbench/components/ui';
import { useRegisterGenerateDraftFlusher } from '@workbench/widgets/generate/generateDraftRegistry';
import { useDebouncedDraftValue } from '@workbench/widgets/generate/useDebouncedDraftValue';
import { useRef, useState } from 'react';

import { PositivePromptActions, PromptTriggerPopover } from './PositivePromptActions';
import { PROMPT_ATTENTION_TARGET_PROPS } from './promptAttentionHotkeys';
import { insertPromptText, type PromptTextRange, registerPositivePromptElement } from './promptFocus';
import { resetPromptHistoryNavigation } from './promptHistoryNavigation';
import { PromptTextarea } from './PromptTextarea';

const PROMPT_INPUT_DEBOUNCE_MS = 250;

interface PositivePromptFieldProps {
  heightPx: number;
  loras: GenerateLora[];
  projectId: string;
  selectedModel: GenerateModelConfig | undefined;
  showSyntaxHighlighting: boolean;
  value: string;
  onChange: (value: string) => void;
  onResizeEnd: (heightPx: number) => void;
  onUsePrompt: (prompt: PromptHistoryItem) => void;
}

type PromptTriggerPickerState = {
  anchorRect: { height: number; width: number; x: number; y: number };
  range?: PromptTextRange;
};

export const PositivePromptField = ({
  heightPx,
  loras,
  onChange,
  onResizeEnd,
  onUsePrompt,
  projectId,
  selectedModel,
  showSyntaxHighlighting,
  value,
}: PositivePromptFieldProps) => {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [triggerPickerState, setTriggerPickerState] = useState<PromptTriggerPickerState | null>(null);
  const { commitDraftValue, draftValue, flushDraftValue, replaceDraftValue, setDraftValue } = useDebouncedDraftValue({
    delayMs: PROMPT_INPUT_DEBOUNCE_MS,
    onCommit: onChange,
    resetKey: projectId,
    value,
  });

  useRegisterGenerateDraftFlusher(flushDraftValue);

  const commitPromptChange = (nextValue: string) => {
    resetPromptHistoryNavigation();
    setDraftValue(nextValue);
  };

  const commitPromptChangeImmediately = (nextValue: string) => {
    resetPromptHistoryNavigation();
    commitDraftValue(nextValue);
  };

  const openPromptTriggerPicker = (anchorElement: HTMLElement, range?: PromptTextRange) => {
    const rect = anchorElement.getBoundingClientRect();

    setTriggerPickerState({
      anchorRect: { height: rect.height, width: rect.width, x: rect.x, y: rect.y },
      range,
    });
  };

  const handlePromptKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== '<' || event.altKey || event.ctrlKey || event.metaKey) {
      return;
    }

    event.preventDefault();
    openPromptTriggerPicker(event.currentTarget, {
      end: event.currentTarget.selectionEnd,
      start: event.currentTarget.selectionStart,
    });
  };

  const closePromptTriggerPicker = () => setTriggerPickerState(null);

  const selectPromptTrigger = (trigger: string) => {
    insertPromptText({
      onChange: commitPromptChange,
      range: triggerPickerState?.range,
      textarea: textareaRef.current,
      text: trigger,
      value: draftValue,
    });
    closePromptTriggerPicker();
  };

  const handleUsePrompt = (prompt: PromptHistoryItem) => {
    replaceDraftValue(prompt.positivePrompt);
    onUsePrompt(prompt);
  };

  const handleTextareaRef = (element: HTMLTextAreaElement | null) => {
    textareaRef.current = element;
    registerPositivePromptElement(element);
  };

  return (
    <Field
      label="Prompt"
      labelEnd={
        <PositivePromptActions
          isPromptTriggerPickerOpen={triggerPickerState !== null}
          loras={loras}
          positivePrompt={draftValue}
          projectId={projectId}
          selectedModel={selectedModel}
          onOpenPromptTriggerPicker={(anchorElement) => openPromptTriggerPicker(anchorElement)}
          onPositivePromptChangeImmediate={commitPromptChangeImmediately}
          onUsePrompt={handleUsePrompt}
        />
      }
    >
      <PromptTextarea
        {...PROMPT_ATTENTION_TARGET_PROPS}
        aria-label="Positive prompt"
        defaultHeightPx={heightPx}
        maxHeightPx={360}
        minHeightPx={96}
        resizeHandleAriaLabel="Resize positive prompt"
        size="xs"
        fontFamily="mono"
        showSyntaxHighlighting={showSyntaxHighlighting}
        textareaRef={handleTextareaRef}
        value={draftValue}
        onChange={(event: ChangeEvent<HTMLTextAreaElement>) => commitPromptChange(event.currentTarget.value)}
        onKeyDown={handlePromptKeyDown}
        onResizeEnd={onResizeEnd}
      />
      {triggerPickerState ? (
        <PromptTriggerPopover
          loras={loras}
          open
          positioning={{ getAnchorRect: () => triggerPickerState.anchorRect }}
          selectedModel={selectedModel}
          onClose={closePromptTriggerPicker}
          onSelect={selectPromptTrigger}
        />
      ) : null}
    </Field>
  );
};
