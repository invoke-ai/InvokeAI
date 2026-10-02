/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { GenerateLora, GenerateModelConfig } from '@features/generation/core/types';
import type { ChangeEvent, KeyboardEvent } from 'react';

import { HStack, Text } from '@chakra-ui/react';
import { getPromptTemplateChunks } from '@features/generation/core/promptTemplates';
import { useRegisterGenerateDraftFlusher } from '@features/generation/ui/generateDraftRegistry';
import { useDebouncedDraftValue } from '@features/generation/ui/useDebouncedDraftValue';
import { Button, Field, IconButton, Tooltip } from '@platform/ui';
import { PlusIcon, XIcon } from 'lucide-react';
import { useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { AddPromptTriggerButton, PromptTriggerPopover } from './PositivePromptActions';
import { PROMPT_ATTENTION_TARGET_PROPS } from './promptAttentionHotkeys';
import { insertPromptText } from './promptFocus';
import { promptHistoryNavigation } from './promptHistoryNavigation';
import { PromptTextarea } from './PromptTextarea';
import { usePromptTriggerAutocomplete } from './usePromptTriggerAutocomplete';
import { usePromptTriggerPicker } from './usePromptTriggerPicker';

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

/** No `__name__` here: a negative prompt is never expanded, so it has no wildcards. */
const NEGATIVE_PROMPT_TRIGGER_KEYS = ['<'] as const;
const NEGATIVE_PROMPT_TRIGGER_KINDS = new Set(['embedding', 'phrase'] as const);

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

  // Preserve textarea identity; read-only requires nonempty negative template content.
  const viewedTemplatePrompt = isTemplateViewMode && templateNegativePrompt ? templateNegativePrompt : null;
  const isViewingMerged = viewedTemplatePrompt !== null;

  const autocomplete = usePromptTriggerAutocomplete({
    isDisabled: isViewingMerged,
    keys: NEGATIVE_PROMPT_TRIGGER_KEYS,
    loras,
    onChange: commitPromptChange,
    selectedModel,
  });

  const insertTrigger = useCallback(
    (trigger: string) => {
      insertPromptText({
        onChange: commitPromptChange,
        textarea: textareaRef.current,
        text: trigger,
        value: draftValue,
      });
    },
    [commitPromptChange, draftValue]
  );
  const triggerPicker = usePromptTriggerPicker({ insert: insertTrigger });

  const handlePromptKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.altKey || event.ctrlKey || event.metaKey) {
        return;
      }

      autocomplete.handleKeyDown(event);
    },
    [autocomplete]
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

  const labelEnd = useMemo(
    () => (
      <HStack gap="0.5">
        <AddPromptTriggerButton
          isOpen={triggerPicker.isOpen || isViewingMerged}
          onOpenPromptTriggerPicker={triggerPicker.open}
        />
        <Tooltip content={t('widgets.generate.negativePromptHide')}>
          <IconButton
            aria-label={t('widgets.generate.negativePromptHide')}
            color="fg.muted"
            size="2xs"
            variant="ghost"
            onClick={() => onEnabledChange(false)}
          >
            <XIcon />
          </IconButton>
        </Tooltip>
      </HStack>
    ),
    [isViewingMerged, onEnabledChange, t, triggerPicker]
  );

  const templateChunks = useMemo(
    () => (viewedTemplatePrompt === null ? null : getPromptTemplateChunks(draftValue, viewedTemplatePrompt)),
    [draftValue, viewedTemplatePrompt]
  );
  const exitViewMode = useCallback(() => onTemplateViewModeChange?.(false), [onTemplateViewModeChange]);

  // Keep collapsed previews discoverable without changing persisted negativePromptEnabled.
  if (!isEnabled) {
    return (
      <Button
        aria-label={t('widgets.generate.enableNegativePrompt')}
        color="fg.muted"
        justifyContent="flex-start"
        size="2xs"
        variant="ghost"
        onClick={() => onEnabledChange(true)}
      >
        <PlusIcon />
        {t('widgets.generate.negativePromptAdd')}
        {value.trim() !== '' ? (
          <Text as="span" color="fg.subtle" fontSize="2xs" fontWeight="normal" minW="0" truncate>
            {value}
          </Text>
        ) : null}
      </Button>
    );
  }

  return (
    <Field hint="negativePrompt" label={t('widgets.generate.negativePrompt')} labelEnd={labelEnd} helpText={helpText}>
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
      {triggerPicker.dismissElement}
      {triggerPicker.isOpen ? (
        <PromptTriggerPopover
          allowedKinds={NEGATIVE_PROMPT_TRIGGER_KINDS}
          loras={loras}
          open
          positioning={triggerPicker.positioning}
          selectedModel={selectedModel}
          onClose={triggerPicker.close}
          onSelect={triggerPicker.select}
        />
      ) : null}
    </Field>
  );
};
