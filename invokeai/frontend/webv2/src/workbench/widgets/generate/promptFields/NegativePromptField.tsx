/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { GenerateLora, GenerateModelConfig } from '@workbench/generation/types';
import type { ChangeEvent, KeyboardEvent } from 'react';

import { HStack, Switch } from '@chakra-ui/react';
import { Field } from '@workbench/components/ui';
import { useRegisterGenerateDraftFlusher } from '@workbench/widgets/generate/generateDraftRegistry';
import { useDebouncedDraftValue } from '@workbench/widgets/generate/useDebouncedDraftValue';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { AddPromptTriggerButton, PromptTriggerPopover } from './PositivePromptActions';
import { PROMPT_ATTENTION_TARGET_PROPS } from './promptAttentionHotkeys';
import { insertPromptText, type PromptTextRange } from './promptFocus';
import { resetPromptHistoryNavigation } from './promptHistoryNavigation';
import { PromptTextarea } from './PromptTextarea';

const PROMPT_INPUT_DEBOUNCE_MS = 250;

interface NegativePromptFieldProps {
  heightPx: number;
  helpText?: string;
  isEnabled: boolean;
  loras: GenerateLora[];
  projectId: string;
  selectedModel: GenerateModelConfig | undefined;
  showSyntaxHighlighting: boolean;
  value: string;
  onChange: (value: string) => void;
  onEnabledChange: (isEnabled: boolean) => void;
  onResizeEnd: (heightPx: number) => void;
}

type PromptTriggerPickerState = {
  anchorRect: { height: number; width: number; x: number; y: number };
  range?: PromptTextRange;
};

export const NegativePromptField = ({
  heightPx,
  helpText,
  isEnabled,
  loras,
  onChange,
  onEnabledChange,
  onResizeEnd,
  projectId,
  selectedModel,
  showSyntaxHighlighting,
  value,
}: NegativePromptFieldProps) => {
  const { t } = useTranslation();
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [triggerPickerState, setTriggerPickerState] = useState<PromptTriggerPickerState | null>(null);
  const { draftValue, flushDraftValue, setDraftValue } = useDebouncedDraftValue({
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

  const handleOpenPromptTriggerPicker = useCallback(
    (anchorElement: HTMLElement) => openPromptTriggerPicker(anchorElement),
    [openPromptTriggerPicker]
  );

  const handleEnabledChange = useCallback(
    (event: { checked: boolean }) => onEnabledChange(event.checked),
    [onEnabledChange]
  );

  const handleTextareaRef = useCallback((element: HTMLTextAreaElement | null) => {
    textareaRef.current = element;
  }, []);

  const handlePromptChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => commitPromptChange(event.currentTarget.value),
    [commitPromptChange]
  );

  const labelEnd = useMemo(
    () => (
      <HStack gap="0.5">
        {isEnabled ? (
          <AddPromptTriggerButton
            isOpen={triggerPickerState !== null}
            onOpenPromptTriggerPicker={handleOpenPromptTriggerPicker}
          />
        ) : null}
        <Switch.Root checked={isEnabled} size="sm" onCheckedChange={handleEnabledChange}>
          <Switch.HiddenInput />
          <Switch.Control _checked={{ bg: 'accent.solid' }}>
            <Switch.Thumb />
          </Switch.Control>
          <Switch.Label srOnly>{t('widgets.generate.enableNegativePrompt')}</Switch.Label>
        </Switch.Root>
      </HStack>
    ),
    [handleEnabledChange, handleOpenPromptTriggerPicker, isEnabled, t, triggerPickerState]
  );

  const triggerPickerPositioning = useMemo(
    () => ({ getAnchorRect: () => triggerPickerState?.anchorRect ?? null }),
    [triggerPickerState]
  );

  return (
    <Field label={t('widgets.generate.negativePrompt')} labelEnd={labelEnd} helpText={isEnabled ? helpText : undefined}>
      {isEnabled ? (
        <>
          <PromptTextarea
            {...PROMPT_ATTENTION_TARGET_PROPS}
            aria-label={t('widgets.generate.negativePrompt')}
            defaultHeightPx={heightPx}
            minHeightPx={56}
            resizeHandleAriaLabel={t('widgets.generate.resizeNegativePrompt')}
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
        </>
      ) : null}
    </Field>
  );
};
