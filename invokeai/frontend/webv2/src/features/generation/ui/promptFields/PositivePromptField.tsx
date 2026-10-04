import type { PromptHistoryItem } from '@features/generation/contracts';
import type { PromptTemplateSnapshot } from '@features/generation/core/promptTemplates';
import type { ExpandPromptSuggestion, GenerateLora, GenerateModelConfig } from '@features/generation/core/types';
import type { ChangeEvent, KeyboardEvent } from 'react';

import { Box, Text } from '@chakra-ui/react';
import { applyPromptTemplate, getPromptTemplateChunks } from '@features/generation/core/promptTemplates';
import { useRegisterGenerateDraftFlusher } from '@features/generation/ui/generateDraftRegistry';
import { useDebouncedDraftValue } from '@features/generation/ui/useDebouncedDraftValue';
import { useWildcards } from '@features/generation/ui/useWildcards';
import { DropZone, Field } from '@platform/ui';
import { useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import type { DynamicPromptsFieldConfig } from './DynamicPromptsPanel';

import {
  PositivePromptActions,
  PromptTriggerPopover,
  type PromptTemplateState,
  type SavedPromptModels,
} from './PositivePromptActions';
import { PROMPT_ATTENTION_TARGET_PROPS } from './promptAttentionHotkeys';
import { insertPromptText, registerPositivePromptElement } from './promptFocus';
import { promptHistoryNavigation } from './promptHistoryNavigation';
import { PromptTextarea } from './PromptTextarea';
import { usePromptImageDrop } from './usePromptImageDrop';
import { usePromptTriggerAutocomplete } from './usePromptTriggerAutocomplete';
import { usePromptTriggerPicker } from './usePromptTriggerPicker';

const PROMPT_INPUT_DEBOUNCE_MS = 250;

interface PositivePromptFieldProps {
  batchCount?: number;
  /** Absent on surfaces whose prompt is not batch-expanded (Upscale). */
  dynamicPrompts?: DynamicPromptsFieldConfig | null;
  /** Absent on surfaces whose model family has no prompt enhancer of its own. */
  expandPromptSuggestion?: ExpandPromptSuggestion | null;
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
  /** Replace the prompt and remove its template atomically; omit this action on non-template surfaces. */
  onFlattenPromptTemplate?: (prompt: string) => void;
  /** Apply or clear the active template. Absent on surfaces with no templates. */
  onApplyPromptTemplate?: (template: PromptTemplateSnapshot | null) => void;
  onResizeEnd: (heightPx: number) => void;
  onUsePrompt: (prompt: PromptHistoryItem) => void;
  /** Where Expand and Image to Prompt save their model picks; absent keeps them for the session. */
  savedPromptModels?: SavedPromptModels;
}

/** The positive prompt is the only field whose `__name__` references resolve. */
const POSITIVE_PROMPT_TRIGGER_KEYS = ['<', '_'] as const;

export const PositivePromptField = ({
  batchCount = 1,
  dynamicPrompts = null,
  expandPromptSuggestion = null,
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
  savedPromptModels,
  selectedModel,
  showSyntaxHighlighting,
  value,
}: PositivePromptFieldProps) => {
  const { t } = useTranslation();
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
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

  // Preserve textarea identity for sizing/hotkeys; only an actual template makes it read-only.
  const isViewingMerged = isTemplateViewMode && promptTemplate !== null;

  // Disable drops while merged view hides authored text. Destructure the ref to keep compiler analysis safe.
  const {
    droppedImage,
    isDragActive: isImageDragActive,
    isOver: isImageDropOver,
    setNodeRef: setImageDropRef,
  } = usePromptImageDrop({ disabled: isViewingMerged });

  const autocomplete = usePromptTriggerAutocomplete({
    isDisabled: isViewingMerged,
    keys: POSITIVE_PROMPT_TRIGGER_KEYS,
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

  const insertTextAtCaret = useCallback(
    (text: string) => {
      insertPromptText({ onChange: commitPromptChange, textarea: textareaRef.current, text, value: draftValue });
    },
    [commitPromptChange, draftValue]
  );

  // Merge against the live draft so expansion counts do not lag behind debounce.
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

  const templateState = useMemo(
    (): PromptTemplateState => ({
      active: promptTemplate,
      isViewMode: isViewingMerged,
      onApply: onApplyPromptTemplate,
      onFlatten: flattenPromptTemplate,
      onViewModeChange: onTemplateViewModeChange,
    }),
    [flattenPromptTemplate, isViewingMerged, onApplyPromptTemplate, onTemplateViewModeChange, promptTemplate]
  );

  const labelEnd = useMemo(
    () => (
      <PositivePromptActions
        batchCount={batchCount}
        droppedImage={droppedImage}
        dynamicPrompts={dynamicPrompts}
        expandPromptSuggestion={expandPromptSuggestion}
        isPromptTriggerPickerOpen={triggerPicker.isOpen}
        showSyntaxHighlighting={showSyntaxHighlighting}
        onInsertText={insertTextAtCaret}
        loras={loras}
        positivePrompt={draftValue}
        effectivePositivePrompt={effectivePositivePrompt}
        template={templateState}
        projectId={projectId}
        savedPromptModels={savedPromptModels}
        selectedModel={selectedModel}
        onOpenPromptTriggerPicker={triggerPicker.open}
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
      expandPromptSuggestion,
      handleUsePrompt,
      droppedImage,
      insertTextAtCaret,
      loras,
      projectId,
      savedPromptModels,
      selectedModel,
      showSyntaxHighlighting,
      templateState,
      triggerPicker,
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

  const templateChunks = useMemo(
    () => (isViewingMerged ? getPromptTemplateChunks(draftValue, promptTemplate.positivePrompt) : null),
    [draftValue, isViewingMerged, promptTemplate]
  );

  /** Clicking the merged text is the way back to editing, as it is in legacy. */
  const exitViewMode = useCallback(() => onTemplateViewModeChange?.(false), [onTemplateViewModeChange]);

  return (
    <Field hint="positivePrompt" label={t('common.prompt')} labelEnd={labelEnd}>
      <Box ref={setImageDropRef} position="relative">
        <PromptTextarea
          {...PROMPT_ATTENTION_TARGET_PROPS}
          {...autocomplete.comboboxProps}
          aria-label={t('widgets.generate.positivePrompt')}
          defaultHeightPx={heightPx}
          minHeightPx={96}
          resizeHandleAriaLabel={t('widgets.generate.resizePositivePrompt')}
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
        {/* Keep pointer events off; dnd-kit targets the wrapper. */}
        {isImageDragActive ? (
          <DropZone
            alignItems="center"
            display="flex"
            inset="0"
            isOver={isImageDropOver}
            justifyContent="center"
            pointerEvents="none"
            position="absolute"
            variant="overlay"
            zIndex="2"
          >
            <Text color="fg" fontSize="lg" fontWeight="700" textAlign="center">
              {t('widgets.generate.dropImageToPrompt')}
            </Text>
          </DropZone>
        ) : null}
      </Box>
      {autocomplete.element}
      {triggerPicker.dismissElement}
      {triggerPicker.isOpen ? (
        <PromptTriggerPopover
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
