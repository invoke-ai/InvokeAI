import type { GenerateLora, GenerateModelConfig } from '@workbench/generation/types';
import type { PromptHistoryItem } from '@workbench/types';
import type { ChangeEvent, KeyboardEvent } from 'react';

import { Field } from '@workbench/components/ui';
import { useRegisterGenerateDraftFlusher } from '@workbench/widgets/generate/generateDraftRegistry';
import { useDebouncedDraftValue } from '@workbench/widgets/generate/useDebouncedDraftValue';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

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
  const { t } = useTranslation();
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [triggerPickerState, setTriggerPickerState] = useState<PromptTriggerPickerState | null>(null);
  const { commitDraftValue, draftValue, flushDraftValue, replaceDraftValue, setDraftValue } = useDebouncedDraftValue({
    delayMs: PROMPT_INPUT_DEBOUNCE_MS,
    onCommit: onChange,
    resetKey: projectId,
    value,
  });

  useRegisterGenerateDraftFlusher(flushDraftValue);

  const commitPromptChange = useCallback(
    (nextValue: string) => {
      resetPromptHistoryNavigation();
      setDraftValue(nextValue);
    },
    [setDraftValue]
  );

  const commitPromptChangeImmediately = useCallback(
    (nextValue: string) => {
      resetPromptHistoryNavigation();
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
      if (event.key !== '<' || event.altKey || event.ctrlKey || event.metaKey) {
        return;
      }

      event.preventDefault();
      openPromptTriggerPicker(event.currentTarget, {
        end: event.currentTarget.selectionEnd,
        start: event.currentTarget.selectionStart,
      });
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

  const labelEnd = useMemo(
    () => (
      <PositivePromptActions
        isPromptTriggerPickerOpen={triggerPickerState !== null}
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
      commitPromptChangeImmediately,
      draftValue,
      handleOpenPromptTriggerPicker,
      handleUsePrompt,
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
