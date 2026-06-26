import type { GenerateLora, GenerateModelConfig } from '@workbench/generation/types';
import type { ModelConfig } from '@workbench/models/types';
import type { PromptHistoryItem } from '@workbench/types';
import type { ChangeEvent } from 'react';

import { HStack, Icon, Image, Input, Popover, Portal, Separator, Stack, Text } from '@chakra-ui/react';
import { getApiErrorMessage } from '@workbench/backend/http';
import { Button, IconButton, Scrollable, Tooltip } from '@workbench/components/ui';
import { filterPromptHistory } from '@workbench/generation/promptHistory';
import { expandPrompt, imageToPrompt } from '@workbench/generation/promptUtilities';
import { getSelectedGalleryImage } from '@workbench/image-actions';
import { ModelSelect } from '@workbench/models/components/ModelSelect';
import { ensureModelsLoaded, useModelsSelector } from '@workbench/models/modelsStore';
import {
  shallowEqual,
  useActiveProjectId,
  useActiveProjectSelector,
  useWorkbenchDispatch,
} from '@workbench/WorkbenchContext';
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
import { useEffect, useId, useRef, useState } from 'react';

const DISABLED_PROMPT_ACTIONS = [
  { icon: BookDashedIcon, label: 'Prompt Template' },
  { icon: CurlyBracesIcon, label: 'Show dynamic prompts' },
];

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
}: PositivePromptActionsProps) => (
  <HStack gap="0.5">
    <AddPromptTriggerButton isOpen={isPromptTriggerPickerOpen} onOpenPromptTriggerPicker={onOpenPromptTriggerPicker} />
    {DISABLED_PROMPT_ACTIONS.map(({ icon: ActionIcon, label }) => (
      <Tooltip key={label} content={label}>
        <IconButton aria-label={label} disabled size="2xs" variant="ghost">
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
  loras,
  models,
  selectedModel,
}: {
  loras: GenerateLora[];
  models: readonly ModelConfig[];
  selectedModel: GenerateModelConfig | undefined;
}): PromptTriggerOption[] => {
  const options: PromptTriggerOption[] = [];

  for (const phrase of getTriggerPhrases(selectedModel)) {
    options.push({ group: selectedModel?.name ?? 'Main model', label: phrase, value: phrase });
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
        options.push({ group: 'Compatible embeddings', label: model.name, value: `<${model.name}>` });
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
  return (
    <Tooltip content="Add prompt trigger">
      <IconButton
        aria-label="Add prompt trigger"
        disabled={isOpen}
        size="2xs"
        variant="ghost"
        onClick={(event) => onOpenPromptTriggerPicker(event.currentTarget)}
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
  const models = useModelsSelector((snapshot) => snapshot.models);
  const [searchTerm, setSearchTerm] = useState('');
  const options = getPromptTriggerOptions({ loras, models, selectedModel });
  const filteredOptions = options.filter((option) => {
    const query = searchTerm.trim().toLowerCase();

    return !query || option.label.toLowerCase().includes(query) || option.group.toLowerCase().includes(query);
  });
  const groupedOptions = filteredOptions.reduce<Array<{ group: string; options: PromptTriggerOption[] }>>(
    (groups, option) => {
      const existingGroup = groups.find((group) => group.group === option.group);

      if (existingGroup) {
        existingGroup.options.push(option);
      } else {
        groups.push({ group: option.group, options: [option] });
      }

      return groups;
    },
    []
  );

  useEffect(() => {
    ensureModelsLoaded();
  }, []);

  return (
    <Popover.Root
      lazyMount
      open={open}
      positioning={{ ...positioning, placement: 'bottom-start' }}
      unmountOnExit
      onOpenChange={(event) => {
        if (event.open) {
          setSearchTerm('');
        } else {
          onClose();
        }
      }}
    >
      <Portal>
        <Popover.Positioner>
          <Popover.Content bg="bg.muted" borderColor="border.emphasized" borderWidth="1px" w="22rem">
            <Popover.Body p="2.5">
              <Stack gap="2" h="18rem">
                <Input
                  aria-label="Search prompt triggers"
                  disabled={options.length === 0}
                  placeholder="Search prompt triggers"
                  size="xs"
                  value={searchTerm}
                  onChange={(event) => setSearchTerm(event.currentTarget.value)}
                />
                <Separator />
                <Scrollable flex="1" label="Prompt trigger options" minH="0">
                  {options.length === 0 ? (
                    <PromptHistoryEmptyText>No prompt triggers available.</PromptHistoryEmptyText>
                  ) : filteredOptions.length === 0 ? (
                    <PromptHistoryEmptyText>No matching triggers.</PromptHistoryEmptyText>
                  ) : (
                    <Stack gap="2">
                      {groupedOptions.map((group) => (
                        <Stack key={group.group} gap="0">
                          <Text color="fg.subtle" fontSize="2xs" fontWeight="700" px="2" textTransform="uppercase">
                            {group.group}
                          </Text>
                          {group.options.map((option, index) => (
                            <Button
                              key={`${option.group}-${option.value}-${index}`}
                              alignItems="start"
                              h="auto"
                              justifyContent="start"
                              px="2"
                              py="1.5"
                              size="xs"
                              variant="ghost"
                              transitionDuration="faster"
                              onClick={() => onSelect(option.value)}
                            >
                              <Text color="fg" fontSize="xs" textAlign="start" wordBreak="break-word">
                                {option.label}
                              </Text>
                            </Button>
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

const ExpandPromptButton = ({
  onPositivePromptChange,
  positivePrompt,
  projectId,
}: {
  positivePrompt: string;
  projectId: string;
  onPositivePromptChange: (prompt: string) => void;
}) => {
  const dispatch = useWorkbenchDispatch();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const activeProjectId = useActiveProjectId();
  const activeProjectIdRef = useRef(activeProjectId);
  const triggerId = useId();
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModelKey, setSelectedModelKey] = useState<string | null>(null);
  const textLlmModels = models.filter((model) => model.type === 'text_llm');
  const selectedModel = selectedModelKey ? textLlmModels.find((model) => model.key === selectedModelKey) : null;

  activeProjectIdRef.current = activeProjectId;

  useEffect(() => {
    ensureModelsLoaded();
  }, []);

  const runExpandPrompt = async () => {
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
      dispatch({ message: getApiErrorMessage(error, 'Could not expand the prompt.'), type: 'recordError' });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Popover.Root
      ids={{ trigger: triggerId }}
      lazyMount
      open={isOpen}
      positioning={{ placement: 'bottom-end' }}
      onOpenChange={(event) => setIsOpen(event.open)}
    >
      <Tooltip
        content={textLlmModels.length === 0 ? 'No Text LLM installed' : 'Expand prompt'}
        ids={{ trigger: triggerId }}
      >
        <Popover.Trigger asChild>
          <IconButton
            aria-label="Expand prompt"
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
                  Expand Prompt
                </Text>
                {textLlmModels.length === 0 ? (
                  <Text color="fg.subtle" fontSize="xs">
                    Install a Text LLM model to expand prompts.
                  </Text>
                ) : (
                  <>
                    <ModelSelect
                      isClearable={false}
                      modelTypes={['text_llm']}
                      placeholder="Select Text LLM"
                      size="xs"
                      value={selectedModelKey}
                      onChange={(model) => setSelectedModelKey(model?.key ?? null)}
                    />
                    <Button
                      disabled={!selectedModel || !positivePrompt.trim()}
                      loading={isLoading}
                      size="xs"
                      onClick={() => void runExpandPrompt()}
                    >
                      Expand
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
  const dispatch = useWorkbenchDispatch();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const selectedImage = useActiveProjectSelector(getSelectedGalleryImage, shallowEqual);
  const activeProjectId = useActiveProjectId();
  const activeProjectIdRef = useRef(activeProjectId);
  const triggerId = useId();
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModelKey, setSelectedModelKey] = useState<string | null>(null);
  const llavaModels = models.filter((model) => model.type === 'llava_onevision');
  const selectedModel = selectedModelKey ? llavaModels.find((model) => model.key === selectedModelKey) : null;

  activeProjectIdRef.current = activeProjectId;

  useEffect(() => {
    ensureModelsLoaded();
  }, []);

  const runImageToPrompt = async () => {
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
      dispatch({
        message: getApiErrorMessage(error, 'Could not generate a prompt from the selected image.'),
        type: 'recordError',
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Popover.Root
      ids={{ trigger: triggerId }}
      lazyMount
      open={isOpen}
      positioning={{ placement: 'bottom-end' }}
      onOpenChange={(event) => setIsOpen(event.open)}
    >
      <Tooltip
        content={llavaModels.length === 0 ? 'No vision model installed' : 'Image to prompt'}
        ids={{ trigger: triggerId }}
      >
        <Popover.Trigger asChild>
          <IconButton aria-label="Image to prompt" disabled={isLoading} size="2xs" variant="ghost">
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
                  Image to Prompt
                </Text>
                {llavaModels.length === 0 ? (
                  <Text color="fg.subtle" fontSize="xs">
                    Install a LLaVA OneVision model to generate prompts from images.
                  </Text>
                ) : (
                  <>
                    <ModelSelect
                      isClearable={false}
                      modelTypes={['llava_onevision']}
                      placeholder="Select vision model"
                      size="xs"
                      value={selectedModelKey}
                      onChange={(model) => setSelectedModelKey(model?.key ?? null)}
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
                        Select an image in Gallery or Preview first.
                      </Text>
                    )}
                    <Button
                      disabled={!selectedImage || !selectedModel}
                      loading={isLoading}
                      size="xs"
                      onClick={() => void runImageToPrompt()}
                    >
                      Generate Prompt
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
  const dispatch = useWorkbenchDispatch();
  const promptHistory = useActiveProjectSelector((project) => project.promptHistory);
  const historyTriggerId = useId();
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const filteredPrompts = filterPromptHistory(promptHistory, searchTerm);

  const onChangeSearchTerm = (event: ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(event.currentTarget.value);
  };

  const usePrompt = (prompt: PromptHistoryItem) => {
    onUsePrompt(prompt);
    setIsOpen(false);
  };

  return (
    <Popover.Root
      ids={{ trigger: historyTriggerId }}
      lazyMount
      open={isOpen}
      positioning={{ placement: 'bottom-end' }}
      onOpenChange={(event) => setIsOpen(event.open)}
    >
      <Tooltip content="Prompt history" ids={{ trigger: historyTriggerId }}>
        <Popover.Trigger asChild>
          <IconButton aria-label="Prompt history" size="2xs" variant="ghost">
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
                    aria-label="Search prompt history"
                    disabled={promptHistory.length === 0}
                    placeholder="Search prompt history"
                    size="xs"
                    value={searchTerm}
                    onChange={onChangeSearchTerm}
                  />
                  <Button
                    disabled={promptHistory.length === 0}
                    size="xs"
                    variant="ghost"
                    onClick={() => dispatch({ type: 'clearPromptHistory' })}
                  >
                    <Icon as={TrashIcon} boxSize="3" />
                    Clear
                  </Button>
                </HStack>
                <Separator />
                <Scrollable flex="1" label="Prompt history entries" minH="0">
                  {promptHistory.length === 0 ? (
                    <PromptHistoryEmptyText>No prompt history yet.</PromptHistoryEmptyText>
                  ) : filteredPrompts.length === 0 ? (
                    <PromptHistoryEmptyText>No matching prompts.</PromptHistoryEmptyText>
                  ) : (
                    <Stack gap="1">
                      {filteredPrompts.map((prompt, index) => (
                        <>
                          <PromptHistoryItemRow
                            key={`${prompt.positivePrompt}-${prompt.negativePrompt ?? ''}-${index}`}
                            prompt={prompt}
                            onUsePrompt={usePrompt}
                          />
                          <Separator />
                        </>
                      ))}
                    </Stack>
                  )}
                </Scrollable>
                <Text color="fg.subtle" fontSize="2xs" textAlign="center">
                  Alt+Up/Down switches between prompts while focused.
                </Text>
              </Stack>
            </Popover.Body>
          </Popover.Content>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};

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
  const dispatch = useWorkbenchDispatch();

  return (
    <HStack align="start" gap="1.5" pr="1">
      <IconButton aria-label="Use prompt" size="2xs" variant="ghost" onClick={() => onUsePrompt(prompt)}>
        <Icon as={Undo2Icon} boxSize="3.5" />
      </IconButton>
      <Stack flex="1" gap="0.5" minW="0">
        {prompt.positivePrompt ? (
          <Text color="fg" fontSize="2xs" wordBreak="break-word">
            <Text as="span" color="fg.subtle" fontWeight="600">
              Prompt:
            </Text>{' '}
            {prompt.positivePrompt}
          </Text>
        ) : null}
        {prompt.negativePrompt ? (
          <Text color="fg" fontSize="2xs" wordBreak="break-word">
            <Text as="span" color="fg.subtle" fontWeight="600">
              Negative:
            </Text>{' '}
            {prompt.negativePrompt}
          </Text>
        ) : null}
      </Stack>

      <IconButton
        aria-label="Delete prompt history item"
        colorPalette="red"
        size="2xs"
        variant="ghost"
        onClick={() => dispatch({ prompt, type: 'removePromptFromHistory' })}
      >
        <Icon as={TrashIcon} boxSize="3.5" />
      </IconButton>
    </HStack>
  );
};
