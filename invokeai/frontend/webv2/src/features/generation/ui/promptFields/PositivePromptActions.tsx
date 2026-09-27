import type { GenerationModelCatalogItem as ModelConfig, PromptHistoryItem } from '@features/generation/contracts';
import type { PromptTemplateSnapshot } from '@features/generation/core/promptTemplates';
import type {
  ExpandPromptSuggestion,
  GenerateLora,
  GenerateModelConfig,
  ImageWithDims,
} from '@features/generation/core/types';
import type { DynamicPromptsFieldConfig } from '@features/generation/ui/promptFields/DynamicPromptsPanel';
import type { DroppedPromptImage } from '@features/generation/ui/promptFields/usePromptImageDrop';
import type { ChangeEvent, MouseEvent } from 'react';

import { Checkbox, HStack, Icon, Image, Input, Popover, Portal, Separator, Stack, Text } from '@chakra-ui/react';
import { galleryImageUrls } from '@features/gallery/utility';
import { filterPromptHistory } from '@features/generation/core/promptHistory';
import { resolveSelectedSystemPromptId } from '@features/generation/core/systemPrompts';
import { llmTaskProgressStore } from '@features/generation/data/llmTaskProgress';
import { expandPrompt, imageToPrompt } from '@features/generation/data/promptUtilities';
import { GenerationModelSelect as ModelSelect, useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { DynamicPromptsButton } from '@features/generation/ui/promptFields/DynamicPromptsButton';
import { LLMTaskProgressDisplay } from '@features/generation/ui/promptFields/LLMTaskProgressDisplay';
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
import { Button, IconButton, PopoverContent, Scrollable, Tooltip } from '@platform/ui';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import {
  EyeIcon,
  EyeOffIcon,
  HistoryIcon,
  ImageUpIcon,
  PencilSparklesIcon,
  PlusIcon,
  TrashIcon,
  Undo2Icon,
} from 'lucide-react';
import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

const POPOVER_POSITIONING_BOTTOM_END = { placement: 'bottom-end' } as const;
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
    <Button alignSelf="start" px="1.5" size="xs" variant="plain" onClick={handleClick}>
      {t('widgets.generate.openModelManager')}
    </Button>
  );
};

export interface PromptTemplateState {
  /** The applied template, or null when none is. */
  active: PromptTemplateSnapshot | null;
  /** Whether the box is showing the merged result rather than the authored text. */
  isViewMode: boolean;
  /** Absent on surfaces with no template concept (Upscale). */
  onApply?: (template: PromptTemplateSnapshot | null) => void;
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
  /** The authored prompt wrapped by the active template — what actually expands. */
  effectivePositivePrompt: string;
  /** Absent on surfaces whose model family has no prompt enhancer of its own. */
  expandPromptSuggestion?: ExpandPromptSuggestion | null;
  loras: GenerateLora[];
  isPromptTriggerPickerOpen: boolean;
  onUsePrompt: (prompt: PromptHistoryItem) => void;
  positivePrompt: string;
  selectedModel: GenerateModelConfig | undefined;
  projectId: string;
  onOpenPromptTriggerPicker: (anchorElement: HTMLElement) => void;
  onPositivePromptChangeImmediate: (prompt: string) => void;
  template: PromptTemplateState;
  onInsertText: (text: string) => void;
  showSyntaxHighlighting: boolean;
}

export const PositivePromptActions = ({
  batchCount,
  droppedImage,
  dynamicPrompts,
  effectivePositivePrompt,
  expandPromptSuggestion,
  isPromptTriggerPickerOpen,
  onInsertText,
  onOpenPromptTriggerPicker,
  onPositivePromptChangeImmediate,
  onUsePrompt,
  positivePrompt,
  projectId,
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
          positivePrompt={effectivePositivePrompt}
          showSyntaxHighlighting={showSyntaxHighlighting}
          onInsertText={onInsertText}
          // Applying merged expansion text must remove its template atomically.
          onUsePrompt={template.active ? template.onFlatten : onPositivePromptChangeImmediate}
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
        suggestion={expandPromptSuggestion ?? null}
        onPositivePromptChange={onPositivePromptChangeImmediate}
      />
      <ImageToPromptButton
        droppedImage={droppedImage}
        isDisabled={template.isViewMode}
        projectId={projectId}
        onPositivePromptChange={onPositivePromptChangeImmediate}
      />
      <PositivePromptHistoryButton onUsePrompt={onUsePrompt} />
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
      <IconButton aria-label={label} aria-pressed={isViewMode} size="2xs" variant="ghost" onClick={handleClick}>
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
        size="2xs"
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
                    size="xs"
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
                            <Text color="fg.subtle" fontSize="2xs" fontWeight="700" px="2" textTransform="uppercase">
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
      <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
        {t('widgets.generate.addPromptTrigger')}
      </Text>
      <Text color="fg.subtle" fontSize="xs">
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
      size="xs"
      transitionDuration="faster"
      variant="ghost"
      onClick={handleClick}
    >
      <Text color="fg" fontSize="xs" textAlign="start" wordBreak="break-word">
        {option.label}
      </Text>
    </Button>
  );
};

const ExpandPromptButton = ({
  isDisabled,
  onPositivePromptChange,
  positivePrompt,
  projectId,
  suggestion,
}: {
  isDisabled: boolean;
  positivePrompt: string;
  projectId: string;
  suggestion: ExpandPromptSuggestion | null;
  onPositivePromptChange: (prompt: string) => void;
}) => {
  const { t } = useTranslation();
  const {
    models: { catalog: models, ensureLoaded: ensureModelsLoaded },
    notifications,
    project: { activeProjectId },
  } = useGenerationUi();
  const activeProjectIdRef = useRef(activeProjectId);
  const triggerId = useId();
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [selectedModelKey, setSelectedModelKey] = useState<string | null>(null);
  const [selectedSystemPromptId, setSelectedSystemPromptId] = useState<string | null>(null);
  // Keyed by image so unticking one frame does not carry over to the next one.
  const [excludedImageName, setExcludedImageName] = useState<string | null>(null);
  const textLlmModels = models.filter((model) => model.type === 'text_llm');
  const suggestedModelKey = suggestion?.modelSource
    ? (textLlmModels.find((model) => model.source === suggestion.modelSource)?.key ?? null)
    : null;
  // An explicit choice wins; otherwise the widget's suggested enhancer, when it is installed.
  const effectiveModelKey = selectedModelKey ?? suggestedModelKey;
  const selectedModel = effectiveModelKey ? textLlmModels.find((model) => model.key === effectiveModelKey) : null;
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

  // eslint-disable-next-line react/refs
  activeProjectIdRef.current = activeProjectId;

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  const runExpandPrompt = useCallback(async () => {
    if (!selectedModel || !positivePrompt.trim()) {
      return;
    }

    const nextTaskId = createUuid();
    setTaskId(nextTaskId);
    setIsLoading(true);

    try {
      const result = await expandPrompt({
        // Forward the selected prompt's optional token cap to expansion.
        image_name: conditioningImage?.image_name,
        max_tokens: selectedSystemPrompt?.maxTokens ?? undefined,
        model_key: selectedModel.key,
        prompt: positivePrompt,
        system_prompt: selectedSystemPrompt?.content ?? null,
        task_id: nextTaskId,
      });

      if (result.expanded_prompt && activeProjectIdRef.current === projectId) {
        onPositivePromptChange(result.expanded_prompt);
      }

      setIsOpen(false);
    } catch (error) {
      notifications.reportError({
        area: 'expand-prompt',
        message: getApiErrorMessage(error, t('widgets.generate.couldNotExpandPrompt')),
        namespace: 'generation',
        projectId,
      });
    } finally {
      llmTaskProgressStore.delete(nextTaskId);
      setTaskId(null);
      setIsLoading(false);
    }
  }, [
    conditioningImage,
    notifications,
    onPositivePromptChange,
    positivePrompt,
    projectId,
    selectedModel,
    selectedSystemPrompt,
    t,
  ]);

  const popoverIds = useMemo(() => ({ trigger: triggerId }), [triggerId]);
  const suggestedImageName = suggestedImage?.image_name ?? null;
  const handleImageIncludedChange = useCallback(
    (event: { checked: boolean | 'indeterminate' }) =>
      setExcludedImageName(event.checked === true ? null : suggestedImageName),
    [suggestedImageName]
  );
  const handleOpenChange = useCallback((event: { open: boolean }) => setIsOpen(event.open), []);
  const handleModelChange = useCallback((model: ModelConfig | null) => setSelectedModelKey(model?.key ?? null), []);
  const handleRunExpandPrompt = useCallback(() => void runExpandPrompt(), [runExpandPrompt]);

  return (
    <Popover.Root
      ids={popoverIds}
      lazyMount
      open={isOpen}
      positioning={POPOVER_POSITIONING_BOTTOM_END}
      onOpenChange={handleOpenChange}
    >
      {/* Always the feature name: the popover explains a missing model and offers the way out. */}
      <Tooltip content={t('widgets.generate.expandPrompt')} ids={popoverIds}>
        <Popover.Trigger asChild>
          <IconButton
            aria-label={t('widgets.generate.expandPrompt')}
            disabled={isDisabled || isLoading}
            size="2xs"
            variant="ghost"
          >
            <PencilSparklesIcon />
          </IconButton>
        </Popover.Trigger>
      </Tooltip>
      <Portal>
        <Popover.Positioner>
          <PopoverContent w="22rem">
            <Popover.Body p="2.5">
              <Stack gap="2.5">
                <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
                  {t('widgets.generate.expandPrompt')}
                </Text>
                {textLlmModels.length === 0 ? (
                  <>
                    <Text color="fg.subtle" fontSize="xs">
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
                      size="xs"
                      value={effectiveModelKey}
                      onChange={handleModelChange}
                    />
                    {suggestion?.modelSource && !suggestedModelKey ? (
                      <>
                        <Text color="fg.subtle" fontSize="xs">
                          {t('widgets.generate.expandSuggestedModelMissing', {
                            model: suggestion.modelName ?? suggestion.modelSource,
                          })}
                        </Text>
                        <OpenModelManagerButton modelType="text_llm" />
                      </>
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
                    <LLMTaskProgressDisplay taskId={taskId} />
                    {positivePrompt.trim() ? null : (
                      <Text color="fg.subtle" fontSize="xs">
                        {t('widgets.generate.enterPromptToExpand')}
                      </Text>
                    )}
                    <Button
                      // The request carries the system prompt's text, so it waits for the list.
                      disabled={!selectedModel || !positivePrompt.trim() || systemPrompts.isLoading}
                      loading={isLoading}
                      size="xs"
                      onClick={handleRunExpandPrompt}
                    >
                      {t('widgets.generate.expand')}
                    </Button>
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
        <Checkbox.Root checked={isIncluded} size="sm" onCheckedChange={onIncludedChange}>
          <Checkbox.HiddenInput />
          <Checkbox.Control />
          <Checkbox.Label fontSize="xs">{t('widgets.generate.expandFromFirstFrame')}</Checkbox.Label>
        </Checkbox.Root>
      ) : (
        <Text color="fg.subtle" fontSize="xs">
          {t('widgets.generate.expandFirstFrameUnreadable')}
        </Text>
      )}
    </HStack>
  );
};

const ImageToPromptButton = ({
  droppedImage,
  isDisabled,
  onPositivePromptChange,
  projectId,
}: {
  droppedImage: DroppedPromptImage;
  isDisabled: boolean;
  projectId: string;
  onPositivePromptChange: (prompt: string) => void;
}) => {
  const { t } = useTranslation();
  const {
    gallery: { selectedImage },
    models: { catalog: models, ensureLoaded: ensureModelsLoaded },
    notifications,
    project: { activeProjectId },
  } = useGenerationUi();
  const activeProjectIdRef = useRef(activeProjectId);
  const triggerId = useId();
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [selectedModelKey, setSelectedModelKey] = useState<string | null>(null);
  const llavaModels = models.filter((model) => model.type === 'llava_onevision');
  const selectedModel = selectedModelKey ? llavaModels.find((model) => model.key === selectedModelKey) : null;
  // Dropped images take precedence over gallery selection.
  const image = droppedImage.image ?? selectedImage;
  const droppedImageName = droppedImage.image?.imageName ?? null;
  const clearDroppedImage = droppedImage.onClear;

  // eslint-disable-next-line react/refs
  activeProjectIdRef.current = activeProjectId;

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  // Key gestures by image name and clear on close so repeated drops work without rerenders reopening the popup.
  useEffect(() => {
    if (droppedImageName !== null) {
      // eslint-disable-next-line react/set-state-in-effect
      setIsOpen(true);
    }
  }, [droppedImageName]);

  // Clear dropped input on every close path; controlled close bypasses onOpenChange.
  const close = useCallback(() => {
    setIsOpen(false);
    clearDroppedImage();
  }, [clearDroppedImage]);

  const runImageToPrompt = useCallback(async () => {
    if (!image || !selectedModel) {
      return;
    }

    const nextTaskId = createUuid();
    setTaskId(nextTaskId);
    setIsLoading(true);

    try {
      const result = await imageToPrompt({
        image_name: image.imageName,
        model_key: selectedModel.key,
        task_id: nextTaskId,
      });

      if (result.prompt && activeProjectIdRef.current === projectId) {
        onPositivePromptChange(result.prompt);
      }

      close();
    } catch (error) {
      notifications.reportError({
        area: 'image-to-prompt',
        message: getApiErrorMessage(error, t('widgets.generate.couldNotGeneratePromptFromImage')),
        namespace: 'generation',
        projectId,
      });
    } finally {
      llmTaskProgressStore.delete(nextTaskId);
      setTaskId(null);
      setIsLoading(false);
    }
  }, [close, image, notifications, onPositivePromptChange, projectId, selectedModel, t]);

  const popoverIds = useMemo(() => ({ trigger: triggerId }), [triggerId]);
  const handleOpenChange = useCallback((event: { open: boolean }) => (event.open ? setIsOpen(true) : close()), [close]);
  const handleModelChange = useCallback((model: ModelConfig | null) => setSelectedModelKey(model?.key ?? null), []);
  const handleRunImageToPrompt = useCallback(() => void runImageToPrompt(), [runImageToPrompt]);

  return (
    <Popover.Root
      ids={popoverIds}
      lazyMount
      open={isOpen}
      positioning={POPOVER_POSITIONING_BOTTOM_END}
      onOpenChange={handleOpenChange}
    >
      <Tooltip content={t('widgets.generate.imageToPrompt')} ids={popoverIds}>
        <Popover.Trigger asChild>
          <IconButton
            aria-label={t('widgets.generate.imageToPrompt')}
            disabled={isDisabled || isLoading}
            size="2xs"
            variant="ghost"
          >
            <ImageUpIcon />
          </IconButton>
        </Popover.Trigger>
      </Tooltip>
      <Portal>
        <Popover.Positioner>
          <PopoverContent w="22rem">
            <Popover.Body p="2.5">
              <Stack gap="2.5">
                <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
                  {t('widgets.generate.imageToPrompt')}
                </Text>
                {llavaModels.length === 0 ? (
                  <>
                    <Text color="fg.subtle" fontSize="xs">
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
                      size="xs"
                      value={selectedModelKey}
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
                        <MiddleTruncate color="fg.subtle" fontSize="xs" text={image.imageName} />
                      </HStack>
                    ) : (
                      <Text color="fg.subtle" fontSize="xs">
                        {t('widgets.generate.selectImageFirst')}
                      </Text>
                    )}
                    <LLMTaskProgressDisplay taskId={taskId} />
                    <Button
                      disabled={!image || !selectedModel}
                      loading={isLoading}
                      size="xs"
                      onClick={handleRunImageToPrompt}
                    >
                      {t('widgets.generate.generatePrompt')}
                    </Button>
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

const PositivePromptHistoryButton = ({ onUsePrompt }: Pick<PositivePromptActionsProps, 'onUsePrompt'>) => {
  const { t } = useTranslation();
  const { clear: clearPromptHistory, items: promptHistory } = useGenerationUi().promptHistory;
  const historyTriggerId = useId();
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const filteredPrompts = filterPromptHistory(promptHistory, searchTerm);
  const popoverIds = useMemo(() => ({ trigger: historyTriggerId }), [historyTriggerId]);
  const handleOpenChange = useCallback((event: { open: boolean }) => setIsOpen(event.open), []);

  const onChangeSearchTerm = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(event.currentTarget.value);
  }, []);

  const usePrompt = useCallback(
    (prompt: PromptHistoryItem) => {
      onUsePrompt(prompt);
      setIsOpen(false);
    },
    [onUsePrompt]
  );

  return (
    <Popover.Root
      ids={popoverIds}
      lazyMount
      open={isOpen}
      positioning={POPOVER_POSITIONING_BOTTOM_END}
      onOpenChange={handleOpenChange}
    >
      <Tooltip content={t('widgets.generate.promptHistory')} ids={popoverIds}>
        <Popover.Trigger asChild>
          <IconButton aria-label={t('widgets.generate.promptHistory')} size="2xs" variant="ghost">
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
                    size="xs"
                    value={searchTerm}
                    onChange={onChangeSearchTerm}
                  />
                  <Button disabled={promptHistory.length === 0} size="xs" variant="ghost" onClick={clearPromptHistory}>
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
                <Text color="fg.subtle" fontSize="2xs" textAlign="center">
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
    <Text color="fg.subtle" fontSize="xs">
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
      <IconButton aria-label={t('widgets.generate.usePrompt')} size="2xs" variant="ghost" onClick={handleUsePrompt}>
        <Icon as={Undo2Icon} boxSize="3.5" />
      </IconButton>
      <Stack flex="1" gap="0.5" minW="0">
        {prompt.positivePrompt ? (
          <Text color="fg" fontSize="2xs" wordBreak="break-word">
            <Text as="span" color="fg.subtle" fontWeight="600">
              {t('common.prompt')}:
            </Text>{' '}
            {prompt.positivePrompt}
          </Text>
        ) : null}
        {prompt.negativePrompt ? (
          <Text color="fg" fontSize="2xs" wordBreak="break-word">
            <Text as="span" color="fg.subtle" fontWeight="600">
              {t('common.negative')}:
            </Text>{' '}
            {prompt.negativePrompt}
          </Text>
        ) : null}
      </Stack>

      <IconButton
        aria-label={t('widgets.generate.deletePromptHistoryItem')}
        colorPalette="red"
        size="2xs"
        variant="ghost"
        onClick={handleDelete}
      >
        <Icon as={TrashIcon} boxSize="3.5" />
      </IconButton>
    </HStack>
  );
};
