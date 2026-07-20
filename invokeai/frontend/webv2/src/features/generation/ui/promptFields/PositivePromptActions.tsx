/* eslint-disable react/react-compiler */
import type { GenerationModelCatalogItem as ModelConfig, PromptHistoryItem } from '@features/generation/contracts';
import type { GenerateLora, GenerateModelConfig } from '@features/generation/core/types';
import type { ChangeEvent, MouseEvent } from 'react';

import { HStack, Icon, Image, Input, Popover, Portal, Separator, Stack, Text } from '@chakra-ui/react';
import { filterPromptHistory } from '@features/generation/core/promptHistory';
import { expandPrompt, imageToPrompt } from '@features/generation/data/promptUtilities';
import { GenerationModelSelect as ModelSelect, useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { useMountEffect } from '@platform/react/useMountEffect';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, IconButton, Scrollable, Tooltip } from '@platform/ui';
import {
  BookDashedIcon,
  CurlyBracesIcon,
  HistoryIcon,
  ImageUpIcon,
  PlusIcon,
  SparklesIcon,
  TrashIcon,
  Undo2Icon,
} from 'lucide-react';
import { useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

const DISABLED_PROMPT_ACTIONS = [
  { icon: BookDashedIcon, labelKey: 'widgets.generate.promptTemplate' },
  { icon: CurlyBracesIcon, labelKey: 'widgets.generate.showDynamicPrompts' },
];
const POPOVER_POSITIONING_BOTTOM_END = { placement: 'bottom-end' } as const;
const TEXT_LLM_MODEL_TYPES = ['text_llm'];
const LLAVA_MODEL_TYPES = ['llava_onevision'];

interface PositivePromptActionsProps {
  loras: GenerateLora[];
  isPromptTriggerPickerOpen: boolean;
  onUsePrompt: (prompt: PromptHistoryItem) => void;
  positivePrompt: string;
  selectedModel: GenerateModelConfig | undefined;
  projectId: string;
  onOpenPromptTriggerPicker: (anchorElement: HTMLElement) => void;
  onPositivePromptChangeImmediate: (prompt: string) => void;
}

export const PositivePromptActions = ({
  isPromptTriggerPickerOpen,
  onOpenPromptTriggerPicker,
  onPositivePromptChangeImmediate,
  onUsePrompt,
  positivePrompt,
  projectId,
}: PositivePromptActionsProps) => {
  const { t } = useTranslation();

  return (
    <HStack gap="0.5">
      <AddPromptTriggerButton
        isOpen={isPromptTriggerPickerOpen}
        onOpenPromptTriggerPicker={onOpenPromptTriggerPicker}
      />
      {DISABLED_PROMPT_ACTIONS.map(({ icon: ActionIcon, labelKey }) => (
        <Tooltip key={labelKey} content={t(labelKey)}>
          <IconButton aria-label={t(labelKey)} disabled size="2xs" variant="ghost">
            <ActionIcon />
          </IconButton>
        </Tooltip>
      ))}
      <ExpandPromptButton
        positivePrompt={positivePrompt}
        projectId={projectId}
        onPositivePromptChange={onPositivePromptChangeImmediate}
      />
      <ImageToPromptButton projectId={projectId} onPositivePromptChange={onPositivePromptChangeImmediate} />
      <PositivePromptHistoryButton onUsePrompt={onUsePrompt} />
    </HStack>
  );
};

type PromptTriggerOption = {
  group: string;
  label: string;
  value: string;
};

const getTriggerPhrases = (model: unknown): string[] => {
  if (!model || typeof model !== 'object') {
    return [];
  }

  const triggerPhrases = (model as { trigger_phrases?: unknown }).trigger_phrases;

  return Array.isArray(triggerPhrases)
    ? triggerPhrases.filter((phrase): phrase is string => typeof phrase === 'string' && phrase.trim().length > 0)
    : [];
};

const getPromptTriggerOptions = ({
  compatibleEmbeddingsLabel,
  loras,
  mainModelLabel,
  models,
  selectedModel,
}: {
  compatibleEmbeddingsLabel: string;
  loras: GenerateLora[];
  mainModelLabel: string;
  models: readonly ModelConfig[];
  selectedModel: GenerateModelConfig | undefined;
}): PromptTriggerOption[] => {
  const options: PromptTriggerOption[] = [];

  for (const phrase of getTriggerPhrases(selectedModel)) {
    options.push({ group: selectedModel?.name ?? mainModelLabel, label: phrase, value: phrase });
  }

  for (const lora of loras) {
    if (!lora.isEnabled) {
      continue;
    }

    for (const phrase of getTriggerPhrases(lora.model)) {
      options.push({ group: lora.model.name, label: phrase, value: phrase });
    }
  }

  if (selectedModel) {
    for (const model of models) {
      if (model.type === 'embedding' && model.base === selectedModel.base) {
        options.push({ group: compatibleEmbeddingsLabel, label: model.name, value: `<${model.name}>` });
      }
    }
  }

  const seen = new Set<string>();

  return options.filter((option) => {
    const key = `${option.group}:${option.value}`;

    if (seen.has(key)) {
      return false;
    }

    seen.add(key);
    return true;
  });
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
  loras,
  onClose,
  onSelect,
  open,
  positioning,
  selectedModel,
}: Pick<PositivePromptActionsProps, 'loras' | 'selectedModel'> & {
  open: boolean;
  positioning: { getAnchorRect: () => { height: number; width: number; x: number; y: number } | null };
  onClose: () => void;
  onSelect: (trigger: string) => void;
}) => {
  const { t } = useTranslation();
  const { catalog: models, ensureLoaded: ensureModelsLoaded } = useGenerationUi().models;
  const [searchTerm, setSearchTerm] = useState('');

  const options = useMemo(
    () =>
      getPromptTriggerOptions({
        compatibleEmbeddingsLabel: t('widgets.generate.compatibleEmbeddings'),
        loras,
        mainModelLabel: t('widgets.generate.mainModel'),
        models,
        selectedModel,
      }),
    [loras, models, selectedModel, t]
  );

  const filteredOptions = useMemo(
    () =>
      options.filter((option) => {
        const query = searchTerm.trim().toLowerCase();

        return !query || option.label.toLowerCase().includes(query) || option.group.toLowerCase().includes(query);
      }),
    [options, searchTerm]
  );

  const groupedOptions = useMemo(
    () =>
      filteredOptions.reduce<Array<{ group: string; options: PromptTriggerOption[] }>>((groups, option) => {
        const existingGroup = groups.find((group) => group.group === option.group);

        if (existingGroup) {
          existingGroup.options.push(option);
        } else {
          groups.push({ group: option.group, options: [option] });
        }

        return groups;
      }, []),
    [filteredOptions]
  );

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
                    <PromptHistoryEmptyText>{t('widgets.generate.noPromptTriggersAvailable')}</PromptHistoryEmptyText>
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
  onPositivePromptChange,
  positivePrompt,
  projectId,
}: {
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
  const textLlmModels = models.filter((model) => model.type === 'text_llm');
  const selectedModel = selectedModelKey ? textLlmModels.find((model) => model.key === selectedModelKey) : null;

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
      const result = await expandPrompt({ model_key: selectedModel.key, prompt: positivePrompt });

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
  }, [notifications, onPositivePromptChange, positivePrompt, projectId, selectedModel, t]);

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
            disabled={isLoading || !positivePrompt.trim()}
            size="2xs"
            variant="ghost"
          >
            <SparklesIcon />
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
                  <Text color="fg.subtle" fontSize="xs">
                    {t('widgets.generate.installTextLlmToExpandPrompts')}
                  </Text>
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
  onPositivePromptChange,
  projectId,
}: {
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

  activeProjectIdRef.current = activeProjectId;

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  const runImageToPrompt = useCallback(async () => {
    if (!selectedImage || !selectedModel) {
      return;
    }

    setIsLoading(true);

    try {
      const result = await imageToPrompt({ image_name: selectedImage.imageName, model_key: selectedModel.key });

      if (result.prompt && activeProjectIdRef.current === projectId) {
        onPositivePromptChange(result.prompt);
      }

      setIsOpen(false);
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
  }, [notifications, onPositivePromptChange, projectId, selectedImage, selectedModel, t]);

  const popoverIds = useMemo(() => ({ trigger: triggerId }), [triggerId]);
  const handleOpenChange = useCallback((event: { open: boolean }) => setIsOpen(event.open), []);
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
          <IconButton aria-label={t('widgets.generate.imageToPrompt')} disabled={isLoading} size="2xs" variant="ghost">
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
                  <Text color="fg.subtle" fontSize="xs">
                    {t('widgets.generate.installVisionModelToGeneratePrompts')}
                  </Text>
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
                    {selectedImage ? (
                      <HStack gap="2">
                        <Image
                          alt={selectedImage.imageName}
                          boxSize="10"
                          flexShrink="0"
                          objectFit="cover"
                          rounded="md"
                          src={selectedImage.thumbnailUrl || selectedImage.imageUrl}
                        />
                        <Text color="fg.subtle" fontSize="xs" truncate>
                          {selectedImage.imageName}
                        </Text>
                      </HStack>
                    ) : (
                      <Text color="fg.subtle" fontSize="xs">
                        {t('widgets.generate.selectImageFirst')}
                      </Text>
                    )}
                    <Button
                      disabled={!selectedImage || !selectedModel}
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
