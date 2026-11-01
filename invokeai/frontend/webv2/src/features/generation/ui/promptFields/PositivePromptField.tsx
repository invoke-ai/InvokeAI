import type { PromptHistoryItem } from '@features/generation/contracts';
import type { PromptTemplateSnapshot } from '@features/generation/core/promptTemplates';
import type { ExpandPromptSuggestion, GenerateLora, GenerateModelConfig } from '@features/generation/core/types';
import type { ChangeEvent, KeyboardEvent } from 'react';

import { Box, Text } from '@chakra-ui/react';
import { supportsPromptAttentionSyntax } from '@features/generation/core/prompt/compel';
import { getEffectivePositivePrompt, getPositivePromptTemplateChunks } from '@features/generation/core/promptTemplates';
import { useRegisterGenerateDraftFlusher } from '@features/generation/ui/generateDraftRegistry';
import { useDebouncedDraftValue } from '@features/generation/ui/useDebouncedDraftValue';
import { useDynamicPrompts } from '@features/generation/ui/useDynamicPrompts';
import { useWildcards } from '@features/generation/ui/useWildcards';
import { DropZone, Field } from '@platform/ui';
import { useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import type { DynamicPromptsFieldConfig } from './DynamicPromptsPanel';
import type { PromptEditor } from './promptEditor';
import type { PromptTemplateChange } from './PromptTemplatesButton';

import { DynamicPromptsNotice } from './DynamicPromptsNotice';
import {
  PositivePromptActions,
  PromptTriggerPopover,
  type PromptTemplateState,
  type SavedPromptModels,
} from './PositivePromptActions';
import { editTextareaText, insertPromptText, registerPositivePromptElement } from './promptFocus';
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
  onApplyPromptTemplate?: (template: PromptTemplateSnapshot | null, change: PromptTemplateChange) => void;
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
  const readsAttentionSyntax = supportsPromptAttentionSyntax(selectedModel?.base);
  const { commitDraftValue, draftValue, flushDraftValue, readDraftValue, setDraftValue } = useDebouncedDraftValue({
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

  /**
   * Programmatic rewrites (Expand, Image to Prompt, history, dynamic prompt picks) go through the browser's editing so
   * Ctrl+Z restores what was there; where that cannot happen (see `editTextareaText`) the value is written directly.
   */
  const writePrompt = useCallback(
    (nextValue: string, { withoutFocus = false }: { withoutFocus?: boolean } = {}) => {
      flushDraftValue();
      promptHistoryNavigation.reset();

      if (!withoutFocus && editTextareaText(textareaRef.current, nextValue)) {
        flushDraftValue();
      } else {
        commitDraftValue(nextValue);
      }
    },
    [commitDraftValue, flushDraftValue]
  );
  // Read by late results and Undo closures, which outlive the render that made them.
  const projectIdRef = useRef(projectId);
  // eslint-disable-next-line react/refs
  projectIdRef.current = projectId;
  const promptEditor = useMemo(
    (): PromptEditor => ({
      element: () => textareaRef.current,
      isShowing: (id) => textareaRef.current?.isConnected === true && projectIdRef.current === id,
      read: readDraftValue,
      write: writePrompt,
    }),
    [readDraftValue, writePrompt]
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

  // The surface applies the whole entry (negative prompt included) from settings that hold the latest typing; the
  // positive prompt then lands as an edit of its own.
  const handleUsePrompt = useCallback(
    (prompt: PromptHistoryItem) => {
      flushDraftValue();
      onUsePrompt(prompt);
      writePrompt(prompt.positivePrompt);
    },
    [flushDraftValue, onUsePrompt, writePrompt]
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
    () => getEffectivePositivePrompt(draftValue, promptTemplate),
    [draftValue, promptTemplate]
  );

  const flattenPromptTemplate = useCallback(
    (prompt: string) => {
      flushDraftValue();
      onFlattenPromptTemplate?.(prompt);
      writePrompt(prompt);
    },
    [flushDraftValue, onFlattenPromptTemplate, writePrompt]
  );
  const expansion = useDynamicPrompts(effectivePositivePrompt, dynamicPrompts);

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
        attentionSyntax={readsAttentionSyntax}
        batchCount={batchCount}
        droppedImage={droppedImage}
        dynamicPrompts={dynamicPrompts}
        expandPromptSuggestion={expandPromptSuggestion}
        expansion={expansion}
        isPromptTriggerPickerOpen={triggerPicker.isOpen}
        showSyntaxHighlighting={showSyntaxHighlighting}
        onInsertText={insertTextAtCaret}
        loras={loras}
        positivePrompt={draftValue}
        promptEditor={promptEditor}
        template={templateState}
        projectId={projectId}
        savedPromptModels={savedPromptModels}
        selectedModel={selectedModel}
        onOpenPromptTriggerPicker={triggerPicker.open}
        onUsePrompt={handleUsePrompt}
      />
    ),
    [
      batchCount,
      draftValue,
      dynamicPrompts,
      expandPromptSuggestion,
      expansion,
      handleUsePrompt,
      droppedImage,
      insertTextAtCaret,
      loras,
      projectId,
      promptEditor,
      readsAttentionSyntax,
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
    () => (isViewingMerged ? getPositivePromptTemplateChunks(draftValue, promptTemplate) : null),
    [draftValue, isViewingMerged, promptTemplate]
  );

  /** Clicking the merged text is the way back to editing, as it is in legacy. */
  const exitViewMode = useCallback(() => onTemplateViewModeChange?.(false), [onTemplateViewModeChange]);

  return (
    <Field hint="positivePrompt" label={t('common.prompt')} labelEnd={labelEnd}>
      <Box ref={setImageDropRef} position="relative">
        <PromptTextarea
          {...autocomplete.comboboxProps}
          aria-label={t('widgets.generate.positivePrompt')}
          attentionSyntax={readsAttentionSyntax ? 'edit' : 'none'}
          defaultHeightPx={heightPx}
          minHeightPx={96}
          resizeHandleAriaLabel={t('widgets.generate.resizePositivePrompt')}
          fontFamily="mono"
          highlightDynamicPrompts={dynamicPrompts?.enabled === true}
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
      {dynamicPrompts ? (
        <DynamicPromptsNotice config={dynamicPrompts} expansion={expansion} prompt={effectivePositivePrompt} />
      ) : null}
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
