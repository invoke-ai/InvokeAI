import type { PromptHistoryItem } from '@features/generation/contracts';
import type { PromptTemplateSnapshot } from '@features/generation/core/promptTemplates';
import type { GenerateLora, GenerateModelConfig } from '@features/generation/core/types';
import type { ChangeEvent, KeyboardEvent } from 'react';

import { applyPromptTemplate, getPromptTemplateChunks } from '@features/generation/core/promptTemplates';
import { useRegisterGenerateDraftFlusher } from '@features/generation/ui/generateDraftRegistry';
import { useDebouncedDraftValue } from '@features/generation/ui/useDebouncedDraftValue';
import { useWildcards } from '@features/generation/ui/useWildcards';
import { Field } from '@platform/ui';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { DynamicPromptsFieldConfig } from './DynamicPromptsPanel';

import { PositivePromptActions, PromptTriggerPopover } from './PositivePromptActions';
import { PROMPT_ATTENTION_TARGET_PROPS } from './promptAttentionHotkeys';
import { insertPromptText, registerPositivePromptElement } from './promptFocus';
import { promptHistoryNavigation } from './promptHistoryNavigation';
import { PromptTextarea } from './PromptTextarea';
import { usePromptTriggerAutocomplete } from './usePromptTriggerAutocomplete';

const PROMPT_INPUT_DEBOUNCE_MS = 250;

interface PositivePromptFieldProps {
  batchCount?: number;
  /** Absent on surfaces whose prompt is not batch-expanded (Upscale). */
  dynamicPrompts?: DynamicPromptsFieldConfig | null;
  /** Absent on surfaces with no template concept (Upscale). */
  promptTemplate?: PromptTemplateSnapshot | null;
  /** Show the merged prompt read-only instead of the authored text. */
  isTemplateViewMode?: boolean;
  onTemplateViewModeChange?: (viewMode: boolean) => void;
  heightPx: number;
  loras: GenerateLora[];
  projectId: string;
  selectedModel: GenerateModelConfig | undefined;
  showSyntaxHighlighting: boolean;
  value: string;
  onChange: (value: string) => void;
  /**
   * Replace the authored prompt with already-merged text and drop the template in
   * one commit. Absent on surfaces with no template concept.
   */
  onFlattenPromptTemplate?: (prompt: string) => void;
  /** Apply or clear the active template. Absent on surfaces with no templates. */
  onApplyPromptTemplate?: (template: PromptTemplateSnapshot | null) => void;
  onResizeEnd: (heightPx: number) => void;
  onUsePrompt: (prompt: PromptHistoryItem) => void;
}

/** The `+` button's browsable list, anchored to the button that opened it. */
type PromptTriggerPickerState = { anchorRect: { height: number; width: number; x: number; y: number } };

/** The positive prompt is the only field whose `__name__` references resolve. */
const POSITIVE_PROMPT_TRIGGER_KEYS = ['<', '_'] as const;

export const PositivePromptField = ({
  batchCount = 1,
  dynamicPrompts = null,
  heightPx,
  isTemplateViewMode = false,
  loras,
  onApplyPromptTemplate,
  onChange,
  onFlattenPromptTemplate,
  onResizeEnd,
  onTemplateViewModeChange,
  onUsePrompt,
  projectId,
  promptTemplate = null,
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

  const openPromptTriggerPicker = useCallback((anchorElement: HTMLElement) => {
    const rect = anchorElement.getBoundingClientRect();

    setTriggerPickerState({ anchorRect: { height: rect.height, width: rect.width, x: rect.x, y: rect.y } });
  }, []);

  const autocomplete = usePromptTriggerAutocomplete({
    isDisabled: isTemplateViewMode,
    keys: POSITIVE_PROMPT_TRIGGER_KEYS,
    loras,
    onChange: commitPromptChange,
    selectedModel,
  });

  const handlePromptKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.altKey || event.ctrlKey || event.metaKey) {
        return;
      }

      // The autocomplete gets first refusal: while its list is open the arrow
      // keys belong to it rather than to prompt history.
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

  // Merged against the *draft*, not the committed value: the template lives in the
  // store but the draft runs 250ms ahead of it, and reading the store here would
  // leave the expansion count a debounce behind what the user is typing.
  const effectivePositivePrompt = useMemo(
    () => (promptTemplate ? applyPromptTemplate(promptTemplate.positivePrompt, draftValue) : draftValue),
    [draftValue, promptTemplate]
  );

  const flattenPromptTemplate = useCallback(
    (prompt: string) => {
      promptHistoryNavigation.reset();
      replaceDraftValue(prompt);
      onFlattenPromptTemplate?.(prompt);
    },
    [onFlattenPromptTemplate, replaceDraftValue]
  );

  const labelEnd = useMemo(
    () => (
      <PositivePromptActions
        batchCount={batchCount}
        dynamicPrompts={dynamicPrompts}
        isPromptTriggerPickerOpen={triggerPickerState !== null}
        showSyntaxHighlighting={showSyntaxHighlighting}
        onInsertText={insertTextAtCaret}
        loras={loras}
        positivePrompt={draftValue}
        activeTemplate={promptTemplate}
        effectivePositivePrompt={effectivePositivePrompt}
        hasPromptTemplate={promptTemplate !== null}
        isTemplateViewMode={isTemplateViewMode}
        onApplyPromptTemplate={onApplyPromptTemplate}
        onTemplateViewModeChange={onTemplateViewModeChange}
        onFlattenPromptTemplate={flattenPromptTemplate}
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
      effectivePositivePrompt,
      flattenPromptTemplate,
      handleOpenPromptTriggerPicker,
      handleUsePrompt,
      insertTextAtCaret,
      isTemplateViewMode,
      loras,
      onApplyPromptTemplate,
      onTemplateViewModeChange,
      projectId,
      promptTemplate,
      selectedModel,
      showSyntaxHighlighting,
      triggerPickerState,
    ]
  );

  const handlePromptChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => {
      commitPromptChange(event.currentTarget.value);
      autocomplete.refresh(event.currentTarget);
    },
    [autocomplete, commitPromptChange]
  );

  /** Clicking moves the caret, which may land in — or out of — a trigger. */
  const handlePromptClick = useCallback(() => autocomplete.refresh(textareaRef.current), [autocomplete]);

  // View mode keeps the same textarea rather than swapping in a preview
  // component: `ResizableTextarea` reads its height once on mount, and the
  // element is also what `focusPositivePrompt` and the attention hotkeys target,
  // so unmounting it would reset the height and break both.
  const isViewingMerged = isTemplateViewMode && promptTemplate !== null;
  const templateChunks = useMemo(
    () => (isViewingMerged ? getPromptTemplateChunks(draftValue, promptTemplate.positivePrompt) : null),
    [draftValue, isViewingMerged, promptTemplate]
  );

  /** Clicking the merged text is the way back to editing, as it is in legacy. */
  const exitViewMode = useCallback(() => onTemplateViewModeChange?.(false), [onTemplateViewModeChange]);

  const triggerPickerPositioning = useMemo(
    () => ({ getAnchorRect: () => triggerPickerState?.anchorRect ?? null }),
    [triggerPickerState]
  );

  return (
    <Field label={t('common.prompt')} labelEnd={labelEnd}>
      <PromptTextarea
        {...PROMPT_ATTENTION_TARGET_PROPS}
        {...autocomplete.comboboxProps}
        aria-label={t('widgets.generate.positivePrompt')}
        defaultHeightPx={heightPx}
        minHeightPx={96}
        resizeHandleAriaLabel={t('widgets.generate.resizePositivePrompt')}
        size="xs"
        fontFamily="mono"
        highlightDynamicPrompts={dynamicPrompts !== null}
        knownWildcards={knownWildcards}
        readOnly={isViewingMerged}
        showSyntaxHighlighting={showSyntaxHighlighting}
        templateChunks={templateChunks}
        textareaRef={handleTextareaRef}
        title={isViewingMerged ? t('widgets.generate.promptTemplates.editAuthored') : undefined}
        value={isViewingMerged ? effectivePositivePrompt : draftValue}
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
    </Field>
  );
};
