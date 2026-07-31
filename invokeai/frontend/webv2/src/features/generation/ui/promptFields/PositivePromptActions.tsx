/* eslint-disable react/react-compiler */
import type { GenerationModelCatalogItem as ModelConfig, PromptHistoryItem } from '@features/generation/contracts';
import type { PromptTemplateSnapshot } from '@features/generation/core/promptTemplates';
import type { GenerateLora, GenerateModelConfig } from '@features/generation/core/types';
import type { DynamicPromptsFieldConfig } from '@features/generation/ui/promptFields/DynamicPromptsPanel';
import type { DroppedPromptImage } from '@features/generation/ui/promptFields/usePromptImageDrop';
import type { ChangeEvent, MouseEvent } from 'react';

import { HStack, Icon, Image, Input, Popover, Portal, Separator, Stack, Text } from '@chakra-ui/react';
import { filterPromptHistory } from '@features/generation/core/promptHistory';
import { resolveSelectedSystemPromptId } from '@features/generation/core/systemPrompts';
import { expandPrompt, imageToPrompt } from '@features/generation/data/promptUtilities';
import { GenerationModelSelect as ModelSelect, useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { DynamicPromptsButton } from '@features/generation/ui/promptFields/DynamicPromptsButton';
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
import { useMountEffect } from '@platform/react/useMountEffect';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, IconButton, Scrollable, Tooltip } from '@platform/ui';
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
/**
 * The way out of an empty state that needs a model installed. Availability is
 * explained inside the popover rather than by disabling the trigger, so the
 * dead end always has a door.
 */
const OpenModelManagerButton = () => {
  const { t } = useTranslation();
  const { openManager } = useGenerationUi().models;

  return (
    <Button alignSelf="start" px="0" size="xs" variant="plain" onClick={openManager}>
      {t('widgets.generate.openModelManager')}
    </Button>
  );
};

/**
 * Everything the prompt actions know about the applied template, in one prop.
 *
 * These arrived as six flat props among nineteen, three of them optional and
 * one — `hasPromptTemplate` — restating `activeTemplate !== null` at every call
 * site. Grouping them says what they are: not six settings but one state, which
 * either exists or does not, and which the buttons around it read to decide
 * whether the authored prompt is theirs to rewrite.
 */
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
          // Picking one expansion writes already-merged text into the authored
          // field, so the template has to come off in the same commit or the next
          // Invoke would wrap it again.
          onUsePrompt={template.active ? template.onFlatten : onPositivePromptChangeImmediate}
        />
      ) : null}
      {/* These three rewrite the authored prompt, which view mode is hiding, so
          they stay out of reach until the user is looking at their own text. */}
      <AddPromptTriggerButton
        isOpen={isPromptTriggerPickerOpen || template.isViewMode}
        onOpenPromptTriggerPicker={onOpenPromptTriggerPicker}
      />
      <ExpandPromptButton
        isDisabled={template.isViewMode}
        positivePrompt={positivePrompt}
        projectId={projectId}
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

/**
 * Picking a template, and switching between the authored text and the merged
 * result. The second button appears only once there is a template to look
 * through — with nothing applied the two views are the same text.
 */
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

/**
 * Swaps the prompt box between the authored text and the merged result. Quiet
 * states only — the icon changes, nothing tints or animates.
 */
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
  const handleClick = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => onOpenPromptTriggerPicker(event.currentTarget),
    [onOpenPromptTriggerPicker]
  );

  return (
    <Tooltip content={t('widgets.generate.addPromptTrigger')}>
      <IconButton
        aria-label={t('widgets.generate.addPromptTrigger')}
        disabled={isOpen}
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
          <Popover.Content bg="bg.muted" borderColor="border.emphasized" borderWidth="1px" w="22rem">
            <Popover.Body p="2.5">
              <Stack gap="2" h="18rem">
                <Input
                  aria-label={t('widgets.generate.searchPromptTriggers')}
                  disabled={options.length === 0}
                  placeholder={t('widgets.generate.searchPromptTriggers')}
                  size="xs"
                  value={searchTerm}
                  onChange={handleSearchChange}
                />
                <Separator />
                <Scrollable flex="1" label={t('widgets.generate.promptTriggerOptions')} minH="0">
                  {options.length === 0 ? (
                    <Stack align="start" gap="1" px="2">
                      <PromptHistoryEmptyText>{t('widgets.generate.noPromptTriggersAvailable')}</PromptHistoryEmptyText>
                      <OpenModelManagerButton />
                    </Stack>
                  ) : filteredOptions.length === 0 ? (
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
            </Popover.Body>
          </Popover.Content>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
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
}: {
  isDisabled: boolean;
  positivePrompt: string;
  projectId: string;
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
  const [selectedModelKey, setSelectedModelKey] = useState<string | null>(null);
  const [selectedSystemPromptId, setSelectedSystemPromptId] = useState<string | null>(null);
  const textLlmModels = models.filter((model) => model.type === 'text_llm');
  const selectedModel = selectedModelKey ? textLlmModels.find((model) => model.key === selectedModelKey) : null;
  // The list is behind the popover, so a closed button has nothing to fetch.
  const systemPrompts = useSystemPrompts({ isEnabled: isOpen });
  // Resolved rather than stored: the id outlives the record it points at, and an unselected
  // picker should still expand with the first available prompt rather than none at all.
  const effectiveSystemPromptId = resolveSelectedSystemPromptId(systemPrompts.prompts, selectedSystemPromptId);
  const selectedSystemPrompt = systemPrompts.prompts.find((prompt) => prompt.id === effectiveSystemPromptId);

  activeProjectIdRef.current = activeProjectId;

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  const runExpandPrompt = useCallback(async () => {
    if (!selectedModel || !positivePrompt.trim()) {
      return;
    }

    setIsLoading(true);

    try {
      const result = await expandPrompt({
        model_key: selectedModel.key,
        prompt: positivePrompt,
        system_prompt: selectedSystemPrompt?.content ?? null,
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
      setIsLoading(false);
    }
  }, [notifications, onPositivePromptChange, positivePrompt, projectId, selectedModel, selectedSystemPrompt, t]);

  const popoverIds = useMemo(() => ({ trigger: triggerId }), [triggerId]);
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
      <Tooltip
        content={
          textLlmModels.length === 0 ? t('widgets.generate.noTextLlmInstalled') : t('widgets.generate.expandPrompt')
        }
        ids={popoverIds}
      >
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
          <Popover.Content bg="bg.muted" borderColor="border.emphasized" borderWidth="1px" w="22rem">
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
                    <OpenModelManagerButton />
                  </>
                ) : (
                  <>
                    <ModelSelect
                      isClearable={false}
                      modelTypes={TEXT_LLM_MODEL_TYPES}
                      placeholder={t('widgets.generate.selectTextLlm')}
                      size="xs"
                      value={selectedModelKey}
                      onChange={handleModelChange}
                    />
                    <SystemPromptsField
                      catalog={systemPrompts}
                      selectedId={effectiveSystemPromptId}
                      onSelect={setSelectedSystemPromptId}
                    />
                    {positivePrompt.trim() ? null : (
                      <Text color="fg.subtle" fontSize="xs">
                        {t('widgets.generate.enterPromptToExpand')}
                      </Text>
                    )}
                    <Button
                      disabled={!selectedModel || !positivePrompt.trim()}
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
          </Popover.Content>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
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
  const [selectedModelKey, setSelectedModelKey] = useState<string | null>(null);
  const llavaModels = models.filter((model) => model.type === 'llava_onevision');
  const selectedModel = selectedModelKey ? llavaModels.find((model) => model.key === selectedModelKey) : null;
  // A drop names the image outright, which is the whole point of dropping one:
  // it is how you describe something other than what the gallery has selected.
  const image = droppedImage.image ?? selectedImage;
  const droppedImageName = droppedImage.image?.imageName ?? null;
  const clearDroppedImage = droppedImage.onClear;

  activeProjectIdRef.current = activeProjectId;

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  // Dropping onto the prompt box is the gesture that opens this popover. Keyed
  // on the name rather than the object so a re-render cannot reopen a popover
  // the user has just dismissed; closing clears the drop, so the same image
  // dropped twice still reads as two separate gestures.
  useEffect(() => {
    if (droppedImageName !== null) {
      setIsOpen(true);
    }
  }, [droppedImageName]);

  // Closing hands the popover back to the gallery selection. Every close route
  // goes through here, including the one after a successful run — a controlled
  // `open` does not fire `onOpenChange` when it is the code that closes it.
  const close = useCallback(() => {
    setIsOpen(false);
    clearDroppedImage();
  }, [clearDroppedImage]);

  const runImageToPrompt = useCallback(async () => {
    if (!image || !selectedModel) {
      return;
    }

    setIsLoading(true);

    try {
      const result = await imageToPrompt({ image_name: image.imageName, model_key: selectedModel.key });

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
      <Tooltip
        content={
          llavaModels.length === 0 ? t('widgets.generate.noVisionModelInstalled') : t('widgets.generate.imageToPrompt')
        }
        ids={popoverIds}
      >
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
          <Popover.Content bg="bg.muted" borderColor="border.emphasized" borderWidth="1px" w="22rem">
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
                    <OpenModelManagerButton />
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
                        <Text color="fg.subtle" fontSize="xs" truncate>
                          {image.imageName}
                        </Text>
                      </HStack>
                    ) : (
                      <Text color="fg.subtle" fontSize="xs">
                        {t('widgets.generate.selectImageFirst')}
                      </Text>
                    )}
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
          </Popover.Content>
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
          <Popover.Content bg="bg.muted" borderColor="border.emphasized" borderWidth="1px" w="24rem">
            <Popover.Body p="2.5">
              <Stack gap="2" h="18rem">
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
          </Popover.Content>
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
