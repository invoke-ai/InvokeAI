import type { PromptHistoryItem } from '@features/generation/contracts';
import type { PromptTemplateSnapshot } from '@features/generation/core/promptTemplates';
import type { GenerateLora, GenerateModelConfig } from '@features/generation/core/types';
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

import { PositivePromptActions, PromptTriggerPopover, type PromptTemplateState } from './PositivePromptActions';
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

  // View mode keeps the same textarea rather than swapping in a preview
  // component: `ResizableTextarea` reads its height once on mount, and the
  // element is also what `focusPositivePrompt` and the attention hotkeys target,
  // so unmounting it would reset the height and break both.
  //
  // Note the `&&`: view mode with no template applied leaves the prompt fully
  // editable, so gating anything on the toggle alone silently disables it while
  // the user is still typing.
  const isViewingMerged = isTemplateViewMode && promptTemplate !== null;

  // View mode hides the authored prompt, and the actions that would rewrite it
  // are out of reach there — so is the drop that opens one of them.
  //
  // Destructured rather than kept whole: passing `imageDrop.setNodeRef` straight
  // to `ref` makes the compiler read the object it came from as a ref too, and
  // every other field of it then counts as a ref access during render.
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
        isPromptTriggerPickerOpen={triggerPicker.isOpen}
        showSyntaxHighlighting={showSyntaxHighlighting}
        onInsertText={insertTextAtCaret}
        loras={loras}
        positivePrompt={draftValue}
        effectivePositivePrompt={effectivePositivePrompt}
        template={templateState}
        projectId={projectId}
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
      handleUsePrompt,
      droppedImage,
      insertTextAtCaret,
      loras,
      projectId,
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
        {/* Only while a compatible drag is in flight — a prompt box wearing a
            permanent dashed border would read as an upload area, not a field.
            `pointerEvents` stays off so the textarea underneath is unaffected;
            dnd-kit hit-tests the wrapper's rect, not this overlay. */}
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
            <Text color="fg" fontSize="sm" fontWeight="700" textAlign="center">
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
