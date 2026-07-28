/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { GenerateLora, GenerateModelConfig } from '@features/generation/core/types';
import type { ChangeEvent, KeyboardEvent } from 'react';

import { HStack, Switch } from '@chakra-ui/react';
import { getPromptTemplateChunks } from '@features/generation/core/promptTemplates';
import { useRegisterGenerateDraftFlusher } from '@features/generation/ui/generateDraftRegistry';
import { useDebouncedDraftValue } from '@features/generation/ui/useDebouncedDraftValue';
import { Field } from '@platform/ui';
import { useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { AddPromptTriggerButton, PromptTriggerPopover } from './PositivePromptActions';
import { PROMPT_ATTENTION_TARGET_PROPS } from './promptAttentionHotkeys';
import { insertPromptText } from './promptFocus';
import { promptHistoryNavigation } from './promptHistoryNavigation';
import { PromptTextarea } from './PromptTextarea';
import { usePromptTriggerAutocomplete } from './usePromptTriggerAutocomplete';

const PROMPT_INPUT_DEBOUNCE_MS = 250;

interface NegativePromptFieldProps {
  heightPx: number;
  /** The active template's negative side, or null when none is applied. */
  templateNegativePrompt?: string | null;
  /** Show the merged prompt read-only instead of the authored text. */
  isTemplateViewMode?: boolean;
  onTemplateViewModeChange?: (viewMode: boolean) => void;
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

/** The `+` button's browsable list, anchored to the button that opened it. */
type PromptTriggerPickerState = { anchorRect: { height: number; width: number; x: number; y: number } };

/** No `__name__` here: a negative prompt is never expanded, so it has no wildcards. */
const NEGATIVE_PROMPT_TRIGGER_KEYS = ['<'] as const;

export const NegativePromptField = ({
  heightPx,
  helpText,
  isEnabled,
  isTemplateViewMode = false,
  loras,
  onChange,
  onEnabledChange,
  onResizeEnd,
  onTemplateViewModeChange,
  projectId,
  selectedModel,
  showSyntaxHighlighting,
  templateNegativePrompt = null,
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
      promptHistoryNavigation.reset();
      setDraftValue(nextValue);
    },
    [setDraftValue]
  );

  const openPromptTriggerPicker = useCallback((anchorElement: HTMLElement) => {
    const rect = anchorElement.getBoundingClientRect();

    setTriggerPickerState({ anchorRect: { height: rect.height, width: rect.width, x: rect.x, y: rect.y } });
  }, []);

  // Same textarea, read-only, as on the positive side — swapping in a different
  // component would reset the resizer's mounted height. The `&&` matters: view
  // mode with no template leaves the prompt editable, so gating on the toggle
  // alone would kill the autocomplete while the user is still typing.
  const isViewingMerged = isTemplateViewMode && templateNegativePrompt !== null;

  const autocomplete = usePromptTriggerAutocomplete({
    isDisabled: isViewingMerged,
    keys: NEGATIVE_PROMPT_TRIGGER_KEYS,
    loras,
    onChange: commitPromptChange,
    selectedModel,
  });

  const handlePromptKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.altKey || event.ctrlKey || event.metaKey) {
        return;
      }

      autocomplete.handleKeyDown(event);
    },
    [autocomplete]
  );

  const closePromptTriggerPicker = useCallback(() => setTriggerPickerState(null), []);

  const selectPromptTrigger = useCallback(
    (trigger: string) => {
      insertPromptText({
        onChange: commitPromptChange,
        textarea: textareaRef.current,
        text: trigger,
        value: draftValue,
      });
      closePromptTriggerPicker();
    },
    [closePromptTriggerPicker, commitPromptChange, draftValue]
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
    (event: ChangeEvent<HTMLTextAreaElement>) => {
      commitPromptChange(event.currentTarget.value);
      autocomplete.refresh(event.currentTarget);
    },
    [autocomplete, commitPromptChange]
  );

  /** Clicking moves the caret, which may land in — or out of — a trigger. */
  const handlePromptClick = useCallback(() => autocomplete.refresh(textareaRef.current), [autocomplete]);

  // The switch shares the Field.Root with the textarea; an explicit id keeps
  // its label bound to its own hidden input instead of the Field's control id.
  const enableSwitchInputId = useId();
  const labelEnd = useMemo(
    () => (
      <HStack gap="0.5">
        {isEnabled ? (
          <AddPromptTriggerButton
            isOpen={triggerPickerState !== null || isTemplateViewMode}
            onOpenPromptTriggerPicker={handleOpenPromptTriggerPicker}
          />
        ) : null}
        <Switch.Root
          checked={isEnabled}
          ids={{ hiddenInput: enableSwitchInputId, label: `${enableSwitchInputId}-label` }}
          size="sm"
          onCheckedChange={handleEnabledChange}
        >
          <Switch.HiddenInput />
          <Switch.Control _checked={{ bg: 'accent.solid' }}>
            <Switch.Thumb />
          </Switch.Control>
          <Switch.Label srOnly>{t('widgets.generate.enableNegativePrompt')}</Switch.Label>
        </Switch.Root>
      </HStack>
    ),
    [
      enableSwitchInputId,
      handleEnabledChange,
      handleOpenPromptTriggerPicker,
      isEnabled,
      isTemplateViewMode,
      t,
      triggerPickerState,
    ]
  );

  const triggerPickerPositioning = useMemo(
    () => ({ getAnchorRect: () => triggerPickerState?.anchorRect ?? null }),
    [triggerPickerState]
  );

  const templateChunks = useMemo(
    () => (isViewingMerged ? getPromptTemplateChunks(draftValue, templateNegativePrompt) : null),
    [draftValue, isViewingMerged, templateNegativePrompt]
  );
  const exitViewMode = useCallback(() => onTemplateViewModeChange?.(false), [onTemplateViewModeChange]);

  return (
    <Field label={t('widgets.generate.negativePrompt')} labelEnd={labelEnd} helpText={isEnabled ? helpText : undefined}>
      {isEnabled ? (
        <>
          <PromptTextarea
            {...PROMPT_ATTENTION_TARGET_PROPS}
            {...autocomplete.comboboxProps}
            aria-label={t('widgets.generate.negativePrompt')}
            defaultHeightPx={heightPx}
            minHeightPx={56}
            resizeHandleAriaLabel={t('widgets.generate.resizeNegativePrompt')}
            size="xs"
            fontFamily="mono"
            readOnly={isViewingMerged}
            showSyntaxHighlighting={showSyntaxHighlighting}
            templateChunks={templateChunks}
            textareaRef={handleTextareaRef}
            title={isViewingMerged ? t('widgets.generate.promptTemplates.editAuthored') : undefined}
            value={templateChunks ? templateChunks.join('') : draftValue}
            onBlur={autocomplete.close}
            onChange={handlePromptChange}
            onClick={isViewingMerged ? exitViewMode : handlePromptClick}
            onKeyDown={handlePromptKeyDown}
            onResizeEnd={onResizeEnd}
          />
          {autocomplete.element}
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
