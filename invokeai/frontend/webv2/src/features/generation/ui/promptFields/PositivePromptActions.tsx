import type { GenerationModelCatalogItem as ModelConfig, PromptHistoryItem } from '@features/generation/contracts';
import type { PromptTemplateSnapshot } from '@features/generation/core/promptTemplates';
import type {
  ExpandPromptSuggestion,
  GenerateLora,
  GenerateModelConfig,
  GenerateSettings,
  ImageWithDims,
} from '@features/generation/core/types';
import type { DynamicPromptsFieldConfig } from '@features/generation/ui/promptFields/DynamicPromptsPanel';
import type { PromptEditor } from '@features/generation/ui/promptFields/promptEditor';
import type { PromptTemplateChange } from '@features/generation/ui/promptFields/PromptTemplatesButton';
import type { DroppedPromptImage } from '@features/generation/ui/promptFields/usePromptImageDrop';
import type { DynamicPromptsExpansion } from '@features/generation/ui/useDynamicPrompts';
import type { ChangeEvent, MouseEvent } from 'react';

import {
  Checkbox,
  HStack,
  Icon,
  Image,
  Input,
  Popover,
  Portal,
  Separator,
  Spinner,
  Stack,
  Text,
} from '@chakra-ui/react';
import { galleryImageUrls } from '@features/gallery/utility';
import { isModelFromRepo } from '@features/generation/core/baseGenerationPolicies';
import { filterPromptHistory } from '@features/generation/core/promptHistory';
import { resolveSelectedSystemPromptId } from '@features/generation/core/systemPrompts';
import { llmTaskProgressStore } from '@features/generation/data/llmTaskProgress';
import { expandPrompt, imageToPrompt } from '@features/generation/data/promptUtilities';
import { getGeneratePromptUndoKey } from '@features/generation/ui/generateUndo';
import { GenerationModelSelect as ModelSelect, useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { DynamicPromptsButton } from '@features/generation/ui/promptFields/DynamicPromptsButton';
import { LLMTaskProgressDisplay } from '@features/generation/ui/promptFields/LLMTaskProgressDisplay';
import { useFocusAfterPromptWrite } from '@features/generation/ui/promptFields/promptEditor';
import { PromptTemplatesButton } from '@features/generation/ui/promptFields/PromptTemplatesButton';
import {
  filterPromptTriggerOptions,
  filterPromptTriggerOptionsByKind,
  groupPromptTriggerOptions,
  usePromptTriggerOptions,
  type PromptTriggerKind,
  type PromptTriggerOption,
} from '@features/generation/ui/promptFields/promptTriggerOptions';
import { SystemPromptsField } from '@features/generation/ui/promptFields/SystemPromptsField';
import { useSystemPrompts } from '@features/generation/ui/promptFields/useSystemPrompts';
import { createUuid } from '@platform/browser/randomUuid';
import { useMountEffect } from '@platform/react/useMountEffect';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, IconButton } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { PopoverContent } from '@platform/ui/Popover';
import { Scrollable } from '@platform/ui/Scrollable';
import { Tooltip } from '@platform/ui/Tooltip';
import {
  ArrowUpLeftIcon,
  EyeIcon,
  EyeOffIcon,
  HistoryIcon,
  ImageUpIcon,
  PencilSparklesIcon,
  PlusIcon,
  TrashIcon,
} from 'lucide-react';
import { useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

const POPOVER_POSITIONING_BOTTOM_END = { placement: 'bottom-end' } as const;
/** Beside the trigger, so the prompt the request rewrites stays in view while it runs. */
const POPOVER_POSITIONING_BESIDE = { placement: 'right-start' } as const;
const TEXT_LLM_MODEL_TYPES = ['text_llm'];
const LLAVA_MODEL_TYPES = ['llava_onevision'];
/** Keep triggers available so missing-model states can explain recovery. */
const OpenModelManagerButton = ({ modelType }: { modelType?: string }) => {
  const { t } = useTranslation();
  const { openManager } = useGenerationUi().models;
  const handleClick = useCallback(
    () => openManager(modelType === undefined ? undefined : { modelType }),
    [modelType, openManager]
  );

  return (
    <Button alignSelf="start" px="1.5" variant="plain" onClick={handleClick}>
      {t('widgets.generate.openModelManager')}
    </Button>
  );
};

type SavedPromptModelKeys = Pick<GenerateSettings, 'expandPromptModelKey' | 'imageToPromptModelKey'>;

/** A surface that keeps the Expand and Image to Prompt model picks; without one they last for the session. */
export interface SavedPromptModels extends SavedPromptModelKeys {
  onChange: (patch: Partial<SavedPromptModelKeys>) => void;
}

/** The picker lists these single-base model types by name, so the first listed is the first by name. */
const getFirstModelByName = (models: readonly ModelConfig[]): ModelConfig | null =>
  models.reduce<ModelConfig | null>(
    (first, model) =>
      first === null || model.name.localeCompare(first.name, undefined, { sensitivity: 'base' }) < 0 ? model : first,
    null
  );

/**
 * The picked model when it is still installed, else the suggested one, else the first listed. A pick is saved
 * to the surface when it keeps one, and only when it differs from what is saved.
 */
const usePromptModelPick = ({
  candidates,
  onSave,
  savedKey,
  suggestedKey = null,
}: {
  candidates: readonly ModelConfig[];
  onSave?: (key: string) => void;
  savedKey?: string | null;
  suggestedKey?: string | null;
}) => {
  const [sessionKey, setSessionKey] = useState<string | null>(null);
  const pickedKey = onSave ? (savedKey ?? null) : sessionKey;
  const findModel = (key: string | null) => (key ? (candidates.find((model) => model.key === key) ?? null) : null);
  const model = findModel(pickedKey) ?? findModel(suggestedKey) ?? getFirstModelByName(candidates);
  const pick = useCallback(
    (next: ModelConfig | null) => {
      if (!next) {
        return;
      }

      if (!onSave) {
        setSessionKey(next.key);
      } else if (next.key !== savedKey) {
        onSave(next.key);
      }
    },
    [onSave, savedKey]
  );

  return { model, pick };
};

export interface PromptTemplateState {
  /** The applied template, or null when none is. */
  active: PromptTemplateSnapshot | null;
  /** Whether the box is showing the merged result rather than the authored text. */
  isViewMode: boolean;
  /** Absent on surfaces with no template concept (Upscale). */
  onApply?: (template: PromptTemplateSnapshot | null, change: PromptTemplateChange) => void;
  /** Absent wherever `onApply` is; view mode has nothing to show without it. */
  onViewModeChange?: (viewMode: boolean) => void;
  /** Writes already-merged text into the authored field and drops the template. */
  onFlatten: (prompt: string) => void;
}

interface PositivePromptActionsProps {
  batchCount: number;
  /** An image dropped on the prompt box, for the image-to-prompt popover to open on. */
  droppedImage: DroppedPromptImage;
  /** Absent on surfaces whose prompt is not batch-expanded (Upscale). */
  dynamicPrompts: DynamicPromptsFieldConfig | null;
  /** Absent on surfaces whose model family has no prompt enhancer of its own. */
  expandPromptSuggestion?: ExpandPromptSuggestion | null;
  /** The expansion of the prompt inside its template (what generates), shared with the notice under the prompt. */
  expansion: DynamicPromptsExpansion;
  loras: GenerateLora[];
  isPromptTriggerPickerOpen: boolean;
  onUsePrompt: (prompt: PromptHistoryItem) => void;
  positivePrompt: string;
  selectedModel: GenerateModelConfig | undefined;
  projectId: string;
  onOpenPromptTriggerPicker: (anchorElement: HTMLElement) => void;
  /** Reads and rewrites the authored prompt, keeping rewrites undoable with Ctrl+Z. */
  promptEditor: PromptEditor;
  savedPromptModels?: SavedPromptModels;
  template: PromptTemplateState;
  onInsertText: (text: string) => void;
  showSyntaxHighlighting: boolean;
  /** Whether the selected model reads Compel attention syntax, the only syntax left in an expanded prompt. */
  attentionSyntax: boolean;
}

export const PositivePromptActions = ({
  attentionSyntax,
  batchCount,
  droppedImage,
  dynamicPrompts,
  expandPromptSuggestion,
  expansion,
  isPromptTriggerPickerOpen,
  onInsertText,
  onOpenPromptTriggerPicker,
  onUsePrompt,
  positivePrompt,
  projectId,
  promptEditor,
  savedPromptModels,
  showSyntaxHighlighting,
  template,
}: PositivePromptActionsProps) => {
  return (
    <HStack gap="0.5">
      <PromptTemplateControls showSyntaxHighlighting={showSyntaxHighlighting} template={template} />
      {dynamicPrompts ? (
        <DynamicPromptsButton
          batchCount={batchCount}
          config={dynamicPrompts}
          expansion={expansion}
          highlightExpandedPrompts={showSyntaxHighlighting && attentionSyntax}
          promptEditor={promptEditor}
          showSyntaxHighlighting={showSyntaxHighlighting}
          onInsertText={onInsertText}
          // Applying merged expansion text must remove its template atomically.
          onUsePrompt={template.active ? template.onFlatten : promptEditor.write}
        />
      ) : null}
      {/* Disable authored-text rewrites while merged view hides the authored text. */}
      <AddPromptTriggerButton
        isOpen={isPromptTriggerPickerOpen || template.isViewMode}
        onOpenPromptTriggerPicker={onOpenPromptTriggerPicker}
      />
      <ExpandPromptButton
        isDisabled={template.isViewMode}
        positivePrompt={positivePrompt}
        projectId={projectId}
        promptEditor={promptEditor}
        savedPromptModels={savedPromptModels}
        suggestion={expandPromptSuggestion ?? null}
      />
      <ImageToPromptButton
        droppedImage={droppedImage}
        isDisabled={template.isViewMode}
        projectId={projectId}
        promptEditor={promptEditor}
        savedPromptModels={savedPromptModels}
      />
      <PositivePromptHistoryButton projectId={projectId} promptEditor={promptEditor} onUsePrompt={onUsePrompt} />
    </HStack>
  );
};

/** Show the view toggle only for an applied template. */
const PromptTemplateControls = ({
  showSyntaxHighlighting,
  template,
}: {
  showSyntaxHighlighting: boolean;
  template: PromptTemplateState;
}) => (
  <>
    {template.onApply ? (
      <PromptTemplatesButton
        activeTemplate={template.active}
        showSyntaxHighlighting={showSyntaxHighlighting}
        onApply={template.onApply}
      />
    ) : null}
    {template.active && template.onViewModeChange ? (
      <TemplateViewModeButton isViewMode={template.isViewMode} onChange={template.onViewModeChange} />
    ) : null}
  </>
);

const TemplateViewModeButton = ({
  isViewMode,
  onChange,
}: {
  isViewMode: boolean;
  onChange: (viewMode: boolean) => void;
}) => {
  const { t } = useTranslation();
  const handleClick = useCallback(() => onChange(!isViewMode), [isViewMode, onChange]);
  const label = isViewMode
    ? t('widgets.generate.promptTemplates.editAuthored')
    : t('widgets.generate.promptTemplates.viewMerged');

  return (
    <Tooltip content={label}>
      <IconButton aria-label={label} aria-pressed={isViewMode} size="sm" variant="ghost" onClick={handleClick}>
        {isViewMode ? <EyeOffIcon /> : <EyeIcon />}
      </IconButton>
    </Tooltip>
  );
};

export const AddPromptTriggerButton = ({
  isOpen,
  onOpenPromptTriggerPicker,
}: {
  isOpen: boolean;
  onOpenPromptTriggerPicker: (anchorElement: HTMLElement) => void;
}) => {
  const { t } = useTranslation();
  // Guard dismiss-then-click so the same gesture cannot immediately reopen the popup.
  const handleClick = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      if (!isOpen) {
        onOpenPromptTriggerPicker(event.currentTarget);
      }
    },
    [isOpen, onOpenPromptTriggerPicker]
  );

  return (
    <Tooltip content={t('widgets.generate.addPromptTrigger')}>
      <IconButton
        aria-expanded={isOpen}
        aria-label={t('widgets.generate.addPromptTrigger')}
        size="sm"
        variant="ghost"
        onClick={handleClick}
      >
        <PlusIcon />
      </IconButton>
    </Tooltip>
  );
};

export const PromptTriggerPopover = ({
  allowedKinds,
  loras,
  onClose,
  onSelect,
  open,
  positioning,
  selectedModel,
}: Pick<PositivePromptActionsProps, 'loras' | 'selectedModel'> & {
  allowedKinds?: ReadonlySet<PromptTriggerKind>;
  open: boolean;
  positioning: { getAnchorRect: () => { height: number; width: number; x: number; y: number } | null };
  onClose: () => void;
  onSelect: (trigger: string) => void;
}) => {
  const { t } = useTranslation();
  const { ensureLoaded: ensureModelsLoaded } = useGenerationUi().models;
  const [searchTerm, setSearchTerm] = useState('');
  const allOptions = usePromptTriggerOptions(loras, selectedModel);
  const options = useMemo(() => filterPromptTriggerOptionsByKind(allOptions, allowedKinds), [allOptions, allowedKinds]);
  const filteredOptions = useMemo(() => filterPromptTriggerOptions(options, searchTerm), [options, searchTerm]);
  const groupedOptions = useMemo(() => groupPromptTriggerOptions(filteredOptions), [filteredOptions]);

  const popoverPositioning = useMemo(() => ({ ...positioning, placement: 'bottom-start' as const }), [positioning]);

  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (event.open) {
        setSearchTerm('');
      } else {
        onClose();
      }
    },
    [onClose]
  );

  const handleSearchChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(event.currentTarget.value);
  }, []);

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  return (
    <Popover.Root lazyMount open={open} positioning={popoverPositioning} unmountOnExit onOpenChange={handleOpenChange}>
      <Portal>
        <Popover.Positioner>
          <PopoverContent w="22rem">
            <Popover.Body p="2.5">
              {options.length === 0 ? (
                <PromptTriggerEmptyState />
              ) : (
                <Stack gap="2" maxH="18rem">
                  <Input
                    aria-label={t('widgets.generate.searchPromptTriggers')}
                    placeholder={t('widgets.generate.searchPromptTriggers')}
                    value={searchTerm}
                    onChange={handleSearchChange}
                  />
                  <Separator />
                  <Scrollable flex="1" label={t('widgets.generate.promptTriggerOptions')} minH="0">
                    {filteredOptions.length === 0 ? (
                      <PromptHistoryEmptyText>{t('widgets.generate.noMatchingTriggers')}</PromptHistoryEmptyText>
                    ) : (
                      <Stack gap="2">
                        {groupedOptions.map((group) => (
                          <Stack key={group.group} gap="0">
                            <Text color="fg.subtle" fontSize="xs" fontWeight="700" px="2" textTransform="uppercase">
                              {group.group}
                            </Text>
                            {group.options.map((option, index) => (
                              <PromptTriggerOptionButton
                                key={`${option.group}-${option.value}-${index}`}
                                onSelect={onSelect}
                                option={option}
                              />
                            ))}
                          </Stack>
                        ))}
                      </Stack>
                    )}
                  </Scrollable>
                </Stack>
              )}
            </Popover.Body>
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};

const PromptTriggerEmptyState = () => {
  const { t } = useTranslation();

  return (
    <Stack align="start" gap="2.5">
      <Text color="fg.subtle" fontSize="xs" fontWeight="700" textTransform="uppercase">
        {t('widgets.generate.addPromptTrigger')}
      </Text>
      <Text color="fg.subtle" fontSize="md">
        {t('widgets.generate.noPromptTriggersAvailable')}
      </Text>
      <OpenModelManagerButton />
    </Stack>
  );
};

const PromptTriggerOptionButton = ({
  onSelect,
  option,
}: {
  onSelect: (trigger: string) => void;
  option: PromptTriggerOption;
}) => {
  const handleClick = useCallback(() => onSelect(option.value), [onSelect, option.value]);

  return (
    <Button
      alignItems="start"
      h="auto"
      justifyContent="start"
      px="2"
      py="1.5"
      transitionDuration="faster"
      variant="ghost"
      onClick={handleClick}
    >
      <Text color="fg" fontSize="md" textAlign="start" wordBreak="break-word">
        {option.label}
      </Text>
    </Button>
  );
};

interface PromptRewrite {
  /** Drops the request: its result never lands. */
  abandon: () => void;
  isRunning: boolean;
  run: (request: (taskId: string) => Promise<string | null | undefined>, onError: (error: unknown) => void) => void;
  taskId: string | null;
}

/** Long enough to recognise a held result in a toast without filling the screen with it. */
const HELD_RESULT_PREVIEW_LENGTH = 280;

/**
 * One Expand or Image to Prompt request at a time; it keeps running when its popover closes. Its result replaces the
 * prompt only while the prompt reads as it did when the request started: as an edit (Ctrl+Z) while the popover is
 * still open, else without taking focus from wherever the user went; an Undo toast covers both. A prompt changed
 * meanwhile is never overwritten: the result is offered in a toast instead. A field that unmounts or moves to another
 * project drops the request, so a late result never lands in the wrong prompt.
 */
const usePromptRewrite = ({
  editor,
  isOpen,
  onLandedOpen,
  projectId,
  title,
}: {
  editor: PromptEditor;
  /** Whether the popover is open, read when the result lands. */
  isOpen: () => boolean;
  /** Closes the popover after a result landed while it was open. */
  onLandedOpen: () => void;
  projectId: string;
  /** The Undo toast's title once a result replaced the prompt. */
  title: string;
}): PromptRewrite => {
  const { t } = useTranslation();
  const { notifications } = useGenerationUi();
  const runIdRef = useRef(0);
  const [taskId, setTaskId] = useState<string | null>(null);

  useMountEffect(() => () => {
    runIdRef.current += 1;
  });

  const abandon = useCallback(() => {
    runIdRef.current += 1;
    setTaskId(null);
  }, []);

  const apply = useCallback(
    (result: string, previous: string, runProjectId: string, asEdit: boolean) => {
      editor.write(result, { withoutFocus: !asEdit });
      notifications.undoable(title, {
        onUndo: () => {
          if (editor.isShowing(runProjectId) && editor.read() === result) {
            editor.write(previous, { withoutFocus: true });
          } else {
            notifications.info(t('widgets.generate.undoStale'));
          }
        },
        projectId: runProjectId,
        supersedes: getGeneratePromptUndoKey(runProjectId),
      });
    },
    [editor, notifications, t, title]
  );

  const offer = useCallback(
    (result: string, runProjectId: string) =>
      notifications.offer(t('widgets.generate.promptChangedWhileRunning'), {
        actions: [
          {
            label: t('widgets.generate.replacePrompt'),
            onClick: () => {
              if (editor.isShowing(runProjectId)) {
                apply(result, editor.read(), runProjectId, false);
              } else {
                notifications.info(t('widgets.generate.promptResultStale'));
              }
            },
          },
          { label: t('common.discard'), onClick: () => undefined },
        ],
        message:
          result.length > HELD_RESULT_PREVIEW_LENGTH ? `${result.slice(0, HELD_RESULT_PREVIEW_LENGTH)}…` : result,
      }),
    [apply, editor, notifications, t]
  );

  const run = useCallback(
    (request: (taskId: string) => Promise<string | null | undefined>, onError: (error: unknown) => void) => {
      runIdRef.current += 1;
      const runId = runIdRef.current;
      const runProjectId = projectId;
      const nextTaskId = createUuid();
      const startingPrompt = editor.read();
      const isCurrent = () => runId === runIdRef.current;

      setTaskId(nextTaskId);

      void request(nextTaskId)
        .then(
          (result) => {
            // Abandoned, or for a project this field no longer shows: dropped.
            if (!isCurrent() || !editor.isShowing(runProjectId)) {
              return;
            }

            const wasOpen = isOpen();

            if (result && editor.read() === startingPrompt) {
              apply(result, startingPrompt, runProjectId, wasOpen);
            } else if (result) {
              offer(result, runProjectId);
            }

            if (wasOpen) {
              onLandedOpen();
            }
          },
          (error: unknown) => {
            if (isCurrent()) {
              onError(error);
            }
          }
        )
        .finally(() => {
          llmTaskProgressStore.delete(nextTaskId);

          if (isCurrent()) {
            setTaskId(null);
          }
        });
    },
    [apply, editor, isOpen, offer, onLandedOpen, projectId]
  );

  return { abandon, isRunning: taskId !== null, run, taskId };
};

/** While a request runs, the explicit way to drop it; closing the popover leaves it running. */
const StopWaitingButton = ({ rewrite }: { rewrite: PromptRewrite }) => {
  const { t } = useTranslation();

  return rewrite.isRunning ? (
    <Button alignSelf="start" px="1.5" variant="plain" onClick={rewrite.abandon}>
      {t('widgets.generate.stopWaiting')}
    </Button>
  ) : null;
};

/**
 * A popover that runs a prompt rewrite. It sits beside its trigger rather than over the prompt, and closes like any
 * popover: the request keeps running (the trigger spins) and its result lands on its own.
 */
const usePromptRewritePopover = ({
  editor,
  onClose,
  projectId,
  title,
}: {
  editor: PromptEditor;
  /** Extra cleanup on every close path. */
  onClose?: () => void;
  projectId: string;
  title: string;
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const isOpenRef = useRef(isOpen);
  const { finalFocusEl, keepFocusAfterWrite } = useFocusAfterPromptWrite(editor);

  // eslint-disable-next-line react/refs
  isOpenRef.current = isOpen;

  const close = useCallback(() => {
    setIsOpen(false);
    onClose?.();
  }, [onClose]);
  const closeAfterWrite = useCallback(() => {
    keepFocusAfterWrite();
    close();
  }, [close, keepFocusAfterWrite]);
  const readIsOpen = useCallback(() => isOpenRef.current, []);
  const rewrite = usePromptRewrite({ editor, isOpen: readIsOpen, onLandedOpen: closeAfterWrite, projectId, title });
  const handleOpenChange = useCallback((event: { open: boolean }) => (event.open ? setIsOpen(true) : close()), [close]);
  const open = useCallback(() => setIsOpen(true), []);

  return {
    isOpen,
    open,
    rewrite,
    rootProps: { finalFocusEl, onOpenChange: handleOpenChange, open: isOpen, positioning: POPOVER_POSITIONING_BESIDE },
  };
};

const ExpandPromptButton = ({
  isDisabled,
  positivePrompt,
  projectId,
  promptEditor,
  savedPromptModels,
  suggestion,
}: {
  isDisabled: boolean;
  positivePrompt: string;
  projectId: string;
  promptEditor: PromptEditor;
  savedPromptModels: SavedPromptModels | undefined;
  suggestion: ExpandPromptSuggestion | null;
}) => {
  const { t } = useTranslation();
  const {
    models: { catalog: models, ensureLoaded: ensureModelsLoaded },
    notifications,
  } = useGenerationUi();
  const triggerId = useId();
  const { isOpen, rewrite, rootProps } = usePromptRewritePopover({
    editor: promptEditor,
    projectId,
    title: t('widgets.generate.promptExpanded'),
  });
  const [selectedSystemPromptId, setSelectedSystemPromptId] = useState<string | null>(null);
  // Keyed by image so unticking one frame does not carry over to the next one.
  const [excludedImageName, setExcludedImageName] = useState<string | null>(null);
  const textLlmModels = models.filter((model) => model.type === 'text_llm');
  const suggestedModelKey = suggestion?.modelSource
    ? (textLlmModels.find((model) =>
        isModelFromRepo(typeof model.source === 'string' ? model.source : null, suggestion.modelSource ?? '')
      )?.key ?? null)
    : null;
  const onSaveModelKey = savedPromptModels?.onChange;
  const saveModelKey = useMemo(
    () => (onSaveModelKey ? (key: string) => onSaveModelKey({ expandPromptModelKey: key }) : undefined),
    [onSaveModelKey]
  );
  const { model: selectedModel, pick: handleModelChange } = usePromptModelPick({
    candidates: textLlmModels,
    onSave: saveModelKey,
    savedKey: savedPromptModels?.expandPromptModelKey,
    suggestedKey: suggestedModelKey,
  });
  const suggestedModel = textLlmModels.find((model) => model.key === suggestedModelKey) ?? null;
  const handleUseSuggestedModel = useCallback(
    () => handleModelChange(suggestedModel),
    [handleModelChange, suggestedModel]
  );
  const suggestedImage = suggestion?.image ?? null;
  const canReadImages = selectedModel?.supports_images === true;
  const isImageIncluded = suggestedImage !== null && excludedImageName !== suggestedImage.image_name;
  const conditioningImage = suggestedImage && canReadImages && isImageIncluded ? suggestedImage : null;
  // The list is behind the popover, so a closed button has nothing to fetch.
  const systemPrompts = useSystemPrompts({ isEnabled: isOpen });
  const suggestedSystemPromptId = conditioningImage ? suggestion?.imageSystemPromptId : suggestion?.systemPromptId;
  // Resolve selection at read time after deletion or without an explicit choice.
  const effectiveSystemPromptId = resolveSelectedSystemPromptId(
    systemPrompts.prompts,
    selectedSystemPromptId ?? suggestedSystemPromptId ?? null
  );
  const selectedSystemPrompt = systemPrompts.prompts.find((prompt) => prompt.id === effectiveSystemPromptId);

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  const { run } = rewrite;
  const handleRunExpandPrompt = useCallback(() => {
    if (!selectedModel || !positivePrompt.trim()) {
      return;
    }

    run(
      async (taskId) =>
        (
          await expandPrompt({
            // Forward the selected prompt's optional token cap to expansion.
            image_name: conditioningImage?.image_name,
            max_tokens: selectedSystemPrompt?.maxTokens ?? undefined,
            model_key: selectedModel.key,
            prompt: positivePrompt,
            system_prompt: selectedSystemPrompt?.content ?? null,
            task_id: taskId,
          })
        ).expanded_prompt,
      (error) =>
        notifications.reportError({
          area: 'expand-prompt',
          message: getApiErrorMessage(error, t('widgets.generate.couldNotExpandPrompt')),
          namespace: 'generation',
          projectId,
        })
    );
  }, [conditioningImage, notifications, positivePrompt, projectId, run, selectedModel, selectedSystemPrompt, t]);

  const popoverIds = useMemo(() => ({ trigger: triggerId }), [triggerId]);
  const suggestedImageName = suggestedImage?.image_name ?? null;
  const handleImageIncludedChange = useCallback(
    (event: { checked: boolean | 'indeterminate' }) =>
      setExcludedImageName(event.checked === true ? null : suggestedImageName),
    [suggestedImageName]
  );

  return (
    <Popover.Root ids={popoverIds} lazyMount {...rootProps}>
      {/* Always the feature name: the popover explains a missing model and offers the way out. */}
      <Tooltip content={t('widgets.generate.expandPrompt')} ids={popoverIds}>
        <Popover.Trigger asChild>
          <IconButton
            aria-busy={rewrite.isRunning || undefined}
            aria-label={t('widgets.generate.expandPrompt')}
            disabled={isDisabled}
            size="sm"
            variant="ghost"
          >
            {rewrite.isRunning ? <Spinner /> : <PencilSparklesIcon />}
          </IconButton>
        </Popover.Trigger>
      </Tooltip>
      <Portal>
        <Popover.Positioner>
          <PopoverContent w="22rem">
            <Popover.Body p="2.5">
              <Stack gap="2.5">
                <Text color="fg.subtle" fontSize="xs" fontWeight="700" textTransform="uppercase">
                  {t('widgets.generate.expandPrompt')}
                </Text>
                {textLlmModels.length === 0 ? (
                  <>
                    <Text color="fg.subtle" fontSize="md">
                      {t('widgets.generate.installTextLlmToExpandPrompts')}
                    </Text>
                    <OpenModelManagerButton modelType="text_llm" />
                  </>
                ) : (
                  <>
                    <ModelSelect
                      isClearable={false}
                      modelTypes={TEXT_LLM_MODEL_TYPES}
                      placeholder={t('widgets.generate.selectTextLlm')}
                      value={selectedModel?.key ?? null}
                      onChange={handleModelChange}
                    />
                    {suggestion?.modelSource && !suggestedModelKey ? (
                      <>
                        <Text color="fg.subtle" fontSize="md">
                          {t('widgets.generate.expandSuggestedModelMissing', {
                            model: suggestion.modelName ?? suggestion.modelSource,
                          })}
                        </Text>
                        <OpenModelManagerButton modelType="text_llm" />
                      </>
                    ) : null}
                    {suggestedModel && selectedModel?.key !== suggestedModel.key ? (
                      <HStack gap="2">
                        <Text color="fg.subtle" fontSize="md">
                          {t('widgets.generate.expandSuggestedModelAvailable', { model: suggestedModel.name })}
                        </Text>
                        <Button
                          aria-label={t('widgets.generate.expandUseSuggestedModelLabel', {
                            model: suggestedModel.name,
                          })}
                          size="xs"
                          variant="outline"
                          onClick={handleUseSuggestedModel}
                        >
                          {t('widgets.generate.expandUseSuggestedModel')}
                        </Button>
                      </HStack>
                    ) : null}
                    <SystemPromptsField
                      catalog={systemPrompts}
                      selectedId={effectiveSystemPromptId}
                      onSelect={setSelectedSystemPromptId}
                    />
                    {suggestedImage && selectedModel ? (
                      <ExpandPromptImageOption
                        canReadImages={canReadImages}
                        image={suggestedImage}
                        isIncluded={isImageIncluded}
                        onIncludedChange={handleImageIncludedChange}
                      />
                    ) : null}
                    <LLMTaskProgressDisplay taskId={rewrite.taskId} />
                    {positivePrompt.trim() ? null : (
                      <Text color="fg.subtle" fontSize="md">
                        {t('widgets.generate.enterPromptToExpand')}
                      </Text>
                    )}
                    <Button
                      // The request carries the system prompt's text, so it waits for the list.
                      disabled={!selectedModel || !positivePrompt.trim() || systemPrompts.isLoading}
                      loading={rewrite.isRunning}
                      onClick={handleRunExpandPrompt}
                    >
                      {t('widgets.generate.expand')}
                    </Button>
                    <StopWaitingButton rewrite={rewrite} />
                  </>
                )}
              </Stack>
            </Popover.Body>
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};

const ExpandPromptImageOption = ({
  canReadImages,
  image,
  isIncluded,
  onIncludedChange,
}: {
  canReadImages: boolean;
  image: ImageWithDims;
  isIncluded: boolean;
  onIncludedChange: (event: { checked: boolean | 'indeterminate' }) => void;
}) => {
  const { t } = useTranslation();

  return (
    <HStack gap="2">
      {/* Decorative: the checkbox label or the note beside it names the frame. */}
      <Image
        alt=""
        boxSize="10"
        flexShrink="0"
        objectFit="cover"
        opacity={canReadImages && isIncluded ? 1 : 0.5}
        rounded="md"
        src={galleryImageUrls.thumbnail(image.image_name)}
      />
      {canReadImages ? (
        <Checkbox.Root checked={isIncluded} onCheckedChange={onIncludedChange}>
          <Checkbox.HiddenInput />
          <Checkbox.Control />
          <Checkbox.Label fontSize="md">{t('widgets.generate.expandFromFirstFrame')}</Checkbox.Label>
        </Checkbox.Root>
      ) : (
        <Text color="fg.subtle" fontSize="md">
          {t('widgets.generate.expandFirstFrameUnreadable')}
        </Text>
      )}
    </HStack>
  );
};

const ImageToPromptButton = ({
  droppedImage,
  isDisabled,
  projectId,
  promptEditor,
  savedPromptModels,
}: {
  droppedImage: DroppedPromptImage;
  isDisabled: boolean;
  projectId: string;
  promptEditor: PromptEditor;
  savedPromptModels: SavedPromptModels | undefined;
}) => {
  const { t } = useTranslation();
  const {
    gallery: { selectedImage },
    models: { catalog: models, ensureLoaded: ensureModelsLoaded },
    notifications,
  } = useGenerationUi();
  const triggerId = useId();
  // Dropped input clears on every close path; a controlled close bypasses onOpenChange.
  const { open, rewrite, rootProps } = usePromptRewritePopover({
    editor: promptEditor,
    onClose: droppedImage.onClear,
    projectId,
    title: t('widgets.generate.promptFromImage'),
  });
  const llavaModels = models.filter((model) => model.type === 'llava_onevision');
  const onSaveModelKey = savedPromptModels?.onChange;
  const saveModelKey = useMemo(
    () => (onSaveModelKey ? (key: string) => onSaveModelKey({ imageToPromptModelKey: key }) : undefined),
    [onSaveModelKey]
  );
  const { model: selectedModel, pick: handleModelChange } = usePromptModelPick({
    candidates: llavaModels,
    onSave: saveModelKey,
    savedKey: savedPromptModels?.imageToPromptModelKey,
  });
  // Dropped images take precedence over gallery selection.
  const image = droppedImage.image ?? selectedImage;
  const droppedImageName = droppedImage.image?.imageName ?? null;
  // Each drop opens the popover once, keyed by image name, so later renders cannot reopen it.
  const [openedForDrop, setOpenedForDrop] = useState<string | null>(null);

  if (droppedImageName !== openedForDrop) {
    setOpenedForDrop(droppedImageName);

    if (droppedImageName !== null) {
      open();
    }
  }

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  const { run } = rewrite;
  const handleRunImageToPrompt = useCallback(() => {
    if (!image || !selectedModel) {
      return;
    }

    run(
      async (taskId) =>
        (await imageToPrompt({ image_name: image.imageName, model_key: selectedModel.key, task_id: taskId })).prompt,
      (error) =>
        notifications.reportError({
          area: 'image-to-prompt',
          message: getApiErrorMessage(error, t('widgets.generate.couldNotGeneratePromptFromImage')),
          namespace: 'generation',
          projectId,
        })
    );
  }, [image, notifications, projectId, run, selectedModel, t]);

  const popoverIds = useMemo(() => ({ trigger: triggerId }), [triggerId]);

  return (
    <Popover.Root ids={popoverIds} lazyMount {...rootProps}>
      <Tooltip content={t('widgets.generate.imageToPrompt')} ids={popoverIds}>
        <Popover.Trigger asChild>
          <IconButton
            aria-busy={rewrite.isRunning || undefined}
            aria-label={t('widgets.generate.imageToPrompt')}
            disabled={isDisabled}
            size="sm"
            variant="ghost"
          >
            {rewrite.isRunning ? <Spinner /> : <ImageUpIcon />}
          </IconButton>
        </Popover.Trigger>
      </Tooltip>
      <Portal>
        <Popover.Positioner>
          <PopoverContent w="22rem">
            <Popover.Body p="2.5">
              <Stack gap="2.5">
                <Text color="fg.subtle" fontSize="xs" fontWeight="700" textTransform="uppercase">
                  {t('widgets.generate.imageToPrompt')}
                </Text>
                {llavaModels.length === 0 ? (
                  <>
                    <Text color="fg.subtle" fontSize="md">
                      {t('widgets.generate.installVisionModelToGeneratePrompts')}
                    </Text>
                    <OpenModelManagerButton modelType="llava_onevision" />
                  </>
                ) : (
                  <>
                    <ModelSelect
                      isClearable={false}
                      modelTypes={LLAVA_MODEL_TYPES}
                      placeholder={t('widgets.generate.selectVisionModel')}
                      value={selectedModel?.key ?? null}
                      onChange={handleModelChange}
                    />
                    {image ? (
                      <HStack gap="2">
                        <Image
                          alt={image.imageName}
                          boxSize="10"
                          flexShrink="0"
                          objectFit="cover"
                          rounded="md"
                          src={image.thumbnailUrl || image.imageUrl}
                        />
                        <MiddleTruncate color="fg.subtle" fontSize="md" text={image.imageName} />
                      </HStack>
                    ) : (
                      <Text color="fg.subtle" fontSize="md">
                        {t('widgets.generate.selectImageFirst')}
                      </Text>
                    )}
                    <LLMTaskProgressDisplay taskId={rewrite.taskId} />
                    <Button
                      disabled={!image || !selectedModel}
                      loading={rewrite.isRunning}
                      onClick={handleRunImageToPrompt}
                    >
                      {t('widgets.generate.generatePrompt')}
                    </Button>
                    <StopWaitingButton rewrite={rewrite} />
                  </>
                )}
              </Stack>
            </Popover.Body>
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};

const PositivePromptHistoryButton = ({
  onUsePrompt,
  projectId,
  promptEditor,
}: Pick<PositivePromptActionsProps, 'onUsePrompt' | 'projectId' | 'promptEditor'>) => {
  const { t } = useTranslation();
  const { notifications, promptHistory: history } = useGenerationUi();
  const promptHistory = history.items;
  const historyTriggerId = useId();
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const filteredPrompts = filterPromptHistory(promptHistory, searchTerm);
  const popoverIds = useMemo(() => ({ trigger: historyTriggerId }), [historyTriggerId]);
  const handleOpenChange = useCallback((event: { open: boolean }) => setIsOpen(event.open), []);
  const { finalFocusEl, keepFocusAfterWrite } = useFocusAfterPromptWrite(promptEditor);

  const onChangeSearchTerm = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(event.currentTarget.value);
  }, []);

  const usePrompt = useCallback(
    (prompt: PromptHistoryItem) => {
      onUsePrompt(prompt);
      keepFocusAfterWrite();
      setIsOpen(false);
    },
    [keepFocusAfterWrite, onUsePrompt]
  );

  // Clearing is undoable, so it does not ask first; the Undo puts every entry back.
  const clearHistory = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      const cleared = promptHistory;

      // The button disables itself; keep focus inside the popover rather than dropping it to the page.
      event.currentTarget.closest<HTMLElement>('[data-scope="popover"][data-part="content"]')?.focus();
      history.clear(projectId);
      notifications.undoable(t('widgets.generate.promptHistoryCleared', { count: cleared.length }), {
        onUndo: () => history.restore(cleared, projectId),
        projectId,
      });
    },
    [history, notifications, projectId, promptHistory, t]
  );

  return (
    <Popover.Root
      finalFocusEl={finalFocusEl}
      ids={popoverIds}
      lazyMount
      open={isOpen}
      positioning={POPOVER_POSITIONING_BOTTOM_END}
      onOpenChange={handleOpenChange}
    >
      <Tooltip content={t('widgets.generate.promptHistory')} ids={popoverIds}>
        <Popover.Trigger asChild>
          <IconButton aria-label={t('widgets.generate.promptHistory')} size="sm" variant="ghost">
            <HistoryIcon />
          </IconButton>
        </Popover.Trigger>
      </Tooltip>
      <Portal>
        <Popover.Positioner>
          <PopoverContent w="24rem">
            <Popover.Body p="2.5">
              <Stack gap="2" maxH="18rem">
                <HStack justify="space-between">
                  <Input
                    aria-label={t('widgets.generate.searchPromptHistory')}
                    disabled={promptHistory.length === 0}
                    placeholder={t('widgets.generate.searchPromptHistory')}
                    value={searchTerm}
                    onChange={onChangeSearchTerm}
                  />
                  <Button disabled={promptHistory.length === 0} variant="ghost" onClick={clearHistory}>
                    <Icon as={TrashIcon} boxSize="3" />
                    {t('common.clear')}
                  </Button>
                </HStack>
                <Separator />
                <Scrollable flex="1" label={t('widgets.generate.promptHistoryEntries')} minH="0">
                  {promptHistory.length === 0 ? (
                    <PromptHistoryEmptyText>{t('widgets.generate.noPromptHistoryYet')}</PromptHistoryEmptyText>
                  ) : filteredPrompts.length === 0 ? (
                    <PromptHistoryEmptyText>{t('widgets.generate.noMatchingPrompts')}</PromptHistoryEmptyText>
                  ) : (
                    <Stack gap="1">
                      {filteredPrompts.map((prompt, index) => (
                        <PromptHistoryItemWithSeparator
                          key={`${prompt.positivePrompt}-${prompt.negativePrompt ?? ''}-${index}`}
                          prompt={prompt}
                          onUsePrompt={usePrompt}
                        />
                      ))}
                    </Stack>
                  )}
                </Scrollable>
                <Text color="fg.subtle" fontSize="xs" textAlign="center">
                  {t('widgets.generate.promptHistoryKeyboardHelp')}
                </Text>
              </Stack>
            </Popover.Body>
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};

const PromptHistoryItemWithSeparator = ({
  onUsePrompt,
  prompt,
}: {
  onUsePrompt: (prompt: PromptHistoryItem) => void;
  prompt: PromptHistoryItem;
}) => (
  <>
    <PromptHistoryItemRow prompt={prompt} onUsePrompt={onUsePrompt} />
    <Separator />
  </>
);

const PromptHistoryEmptyText = ({ children }: { children: string }) => (
  <HStack h="full" justify="center" minH="9rem">
    <Text color="fg.subtle" fontSize="md">
      {children}
    </Text>
  </HStack>
);

const PromptHistoryItemRow = ({
  onUsePrompt,
  prompt,
}: {
  onUsePrompt: (prompt: PromptHistoryItem) => void;
  prompt: PromptHistoryItem;
}) => {
  const { t } = useTranslation();
  const { remove: removePromptFromHistory } = useGenerationUi().promptHistory;
  const handleUsePrompt = useCallback(() => onUsePrompt(prompt), [onUsePrompt, prompt]);
  const handleDelete = useCallback(() => removePromptFromHistory(prompt), [prompt, removePromptFromHistory]);

  return (
    <HStack align="start" gap="1.5" pr="1">
      <Tooltip content={t('widgets.generate.usePrompt')}>
        <IconButton aria-label={t('widgets.generate.usePrompt')} size="sm" variant="ghost" onClick={handleUsePrompt}>
          <Icon as={ArrowUpLeftIcon} boxSize="3.5" />
        </IconButton>
      </Tooltip>
      <Stack flex="1" gap="0.5" minW="0">
        {prompt.positivePrompt ? (
          <Text color="fg" fontSize="xs" wordBreak="break-word">
            <Text as="span" color="fg.subtle" fontWeight="600">
              {t('common.prompt')}:
            </Text>{' '}
            {prompt.positivePrompt}
          </Text>
        ) : null}
        {prompt.negativePrompt ? (
          <Text color="fg" fontSize="xs" wordBreak="break-word">
            <Text as="span" color="fg.subtle" fontWeight="600">
              {t('common.negative')}:
            </Text>{' '}
            {prompt.negativePrompt}
          </Text>
        ) : null}
      </Stack>

      <Tooltip content={t('widgets.generate.deletePromptHistoryItem')}>
        <IconButton
          aria-label={t('widgets.generate.deletePromptHistoryItem')}
          colorPalette="red"
          size="sm"
          variant="ghost"
          onClick={handleDelete}
        >
          <Icon as={TrashIcon} boxSize="3.5" />
        </IconButton>
      </Tooltip>
    </HStack>
  );
};
