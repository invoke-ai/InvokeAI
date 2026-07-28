import type { PromptHistoryItem } from '@features/generation/contracts';
import type { GenerateLora, GenerateModelConfig } from '@features/generation/core/types';
import type { ChangeEvent, KeyboardEvent } from 'react';

import { useRegisterGenerateDraftFlusher } from '@features/generation/ui/generateDraftRegistry';
import { useDebouncedDraftValue } from '@features/generation/ui/useDebouncedDraftValue';
import { useWildcards } from '@features/generation/ui/useWildcards';
import { Field } from '@platform/ui';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { DynamicPromptsFieldConfig } from './DynamicPromptsPanel';

import { PositivePromptActions, PromptTriggerPopover } from './PositivePromptActions';
import { PROMPT_ATTENTION_TARGET_PROPS } from './promptAttentionHotkeys';
import { insertPromptText, type PromptTextRange, registerPositivePromptElement } from './promptFocus';
import { promptHistoryNavigation } from './promptHistoryNavigation';
import { PromptTextarea } from './PromptTextarea';

const PROMPT_INPUT_DEBOUNCE_MS = 250;

interface PositivePromptFieldProps {
  batchCount?: number;
  /** Absent on surfaces whose prompt is not batch-expanded (Upscale). */
  dynamicPrompts?: DynamicPromptsFieldConfig | null;
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
  batchCount = 1,
  dynamicPrompts = null,
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
  const { t } = useTranslation();
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [triggerPickerState, setTriggerPickerState] = useState<PromptTriggerPickerState | null>(null);
  const { knownNames: knownWildcards } = useWildcards();
  const { commitDraftValue, draftValue, flushDraftValue, replaceDraftValue, setDraftValue } = useDebouncedDraftValue({
    delayMs: PROMPT_INPUT_DEBOUNCE_MS,
    onCommit: onChange,
    resetKey: projectId,
    value,
  });

  useRegisterGenerateDraftFlusher(flushDraftValue);

  const commitPromptChange = useCallback(
    (nextValue: string) => {
      promptHistoryNavigation.reset();
      setDraftValue(nextValue);
    },
    [setDraftValue]
  );

  const commitPromptChangeImmediately = useCallback(
    (nextValue: string) => {
      promptHistoryNavigation.reset();
      commitDraftValue(nextValue);
    },
    [commitDraftValue]
  );

  const openPromptTriggerPicker = useCallback((anchorElement: HTMLElement, range?: PromptTextRange) => {
    const rect = anchorElement.getBoundingClientRect();

    setTriggerPickerState({
      anchorRect: { height: rect.height, width: rect.width, x: rect.x, y: rect.y },
      range,
    });
  }, []);

  const handlePromptKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.altKey || event.ctrlKey || event.metaKey) {
        return;
      }

      // `<` opens the picker outright; `__` does too, on the second underscore,
      // since that is the point where the user has committed to a wildcard.
      const opensPicker =
        event.key === '<' ||
        (event.key === '_' && event.currentTarget.value.slice(0, event.currentTarget.selectionStart).endsWith('_'));

      if (!opensPicker) {
        return;
      }

      event.preventDefault();
      // The `__` case replaces the underscore already typed, so the inserted
      // `__name__` does not end up with three leading underscores.
      const start = event.key === '_' ? event.currentTarget.selectionStart - 1 : event.currentTarget.selectionStart;

      openPromptTriggerPicker(event.currentTarget, { end: event.currentTarget.selectionEnd, start });
    },
    [openPromptTriggerPicker]
  );

  const closePromptTriggerPicker = useCallback(() => setTriggerPickerState(null), []);

  const selectPromptTrigger = useCallback(
    (trigger: string) => {
      insertPromptText({
        onChange: commitPromptChange,
        range: triggerPickerState?.range,
        textarea: textareaRef.current,
        text: trigger,
        value: draftValue,
      });
      closePromptTriggerPicker();
    },
    [closePromptTriggerPicker, commitPromptChange, draftValue, triggerPickerState?.range]
  );

  const handleUsePrompt = useCallback(
    (prompt: PromptHistoryItem) => {
      replaceDraftValue(prompt.positivePrompt);
      onUsePrompt(prompt);
    },
    [onUsePrompt, replaceDraftValue]
  );

  const handleTextareaRef = useCallback((element: HTMLTextAreaElement | null) => {
    textareaRef.current = element;
    registerPositivePromptElement(element);
  }, []);

  const handleOpenPromptTriggerPicker = useCallback(
    (anchorElement: HTMLElement) => openPromptTriggerPicker(anchorElement),
    [openPromptTriggerPicker]
  );

  const insertTextAtCaret = useCallback(
    (text: string) => {
      insertPromptText({ onChange: commitPromptChange, textarea: textareaRef.current, text, value: draftValue });
    },
    [commitPromptChange, draftValue]
  );

  const labelEnd = useMemo(
    () => (
      <PositivePromptActions
        batchCount={batchCount}
        dynamicPrompts={dynamicPrompts}
        isPromptTriggerPickerOpen={triggerPickerState !== null}
        onInsertText={insertTextAtCaret}
        loras={loras}
        positivePrompt={draftValue}
        projectId={projectId}
        selectedModel={selectedModel}
        onOpenPromptTriggerPicker={handleOpenPromptTriggerPicker}
        onPositivePromptChangeImmediate={commitPromptChangeImmediately}
        onUsePrompt={handleUsePrompt}
      />
    ),
    [
      batchCount,
      commitPromptChangeImmediately,
      draftValue,
      dynamicPrompts,
      handleOpenPromptTriggerPicker,
      handleUsePrompt,
      insertTextAtCaret,
      loras,
      projectId,
      selectedModel,
      triggerPickerState,
    ]
  );

  const handlePromptChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => commitPromptChange(event.currentTarget.value),
    [commitPromptChange]
  );

  const triggerPickerPositioning = useMemo(
    () => ({ getAnchorRect: () => triggerPickerState?.anchorRect ?? null }),
    [triggerPickerState]
  );

  return (
    <Field label={t('common.prompt')} labelEnd={labelEnd}>
      <PromptTextarea
        {...PROMPT_ATTENTION_TARGET_PROPS}
        aria-label={t('widgets.generate.positivePrompt')}
        defaultHeightPx={heightPx}
        minHeightPx={96}
        resizeHandleAriaLabel={t('widgets.generate.resizePositivePrompt')}
        size="xs"
        fontFamily="mono"
        highlightDynamicPrompts={dynamicPrompts !== null}
        knownWildcards={knownWildcards}
        showSyntaxHighlighting={showSyntaxHighlighting}
        textareaRef={handleTextareaRef}
        value={draftValue}
        onChange={handlePromptChange}
        onKeyDown={handlePromptKeyDown}
        onResizeEnd={onResizeEnd}
      />
      {triggerPickerState ? (
        <PromptTriggerPopover
          loras={loras}
          open
          positioning={triggerPickerPositioning}
          selectedModel={selectedModel}
          onClose={closePromptTriggerPicker}
          onSelect={selectPromptTrigger}
        />
      ) : null}
    </Field>
  );
};
