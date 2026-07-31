import type { ComboboxOnChange, ComboboxOption, SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Button,
  Combobox,
  Flex,
  FormControl,
  FormLabel,
  IconButton,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Portal,
  spinAnimation,
  Text,
  Tooltip,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useDisclosure } from 'common/hooks/useBoolean';
import { positivePromptChanged, selectPositivePrompt } from 'features/controlLayers/store/paramsSlice';
import { setInstallModelsTabByName } from 'features/modelManagerV2/store/installModelsStore';
import { ModelPicker } from 'features/parameters/components/ModelPicker';
import { setPromptUndo } from 'features/prompt/promptUndo';
import {
  selectedModelKeyChanged,
  selectedSystemPromptIdChanged,
  selectSelectedModelKey,
  selectSelectedSystemPromptId,
} from 'features/prompt/store/expandPromptSlice';
import { openSystemPromptsModal } from 'features/systemPrompts/store/systemPromptModal';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { memo, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPencilSimpleBold, PiSparkleBold } from 'react-icons/pi';
import { useListSystemPromptsQuery } from 'services/api/endpoints/systemPrompts';
import { useExpandPromptMutation } from 'services/api/endpoints/utilities';
import { useTextLLMModels } from 'services/api/hooks/modelsByType';
import type { AnyModelConfig } from 'services/api/types';

const loadingStyles: SystemStyleObject = {
  svg: { animation: spinAnimation },
};

export const ExpandPromptButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const prompt = useAppSelector(selectPositivePrompt);
  const selectedSystemPromptId = useAppSelector(selectSelectedSystemPromptId);
  const selectedModelKey = useAppSelector(selectSelectedModelKey);
  const [modelConfigs] = useTextLLMModels();
  const popover = useDisclosure(false);
  const { data: systemPrompts } = useListSystemPromptsQuery();
  const [expandPrompt, { isLoading }] = useExpandPromptMutation();

  const hasModels = modelConfigs.length > 0;

  const selectedModel = useMemo<AnyModelConfig | undefined>(
    () => modelConfigs.find((m) => m.key === selectedModelKey),
    [modelConfigs, selectedModelKey]
  );

  const selectedSystemPrompt = useMemo(
    () => systemPrompts?.find((p) => p.id === selectedSystemPromptId),
    [systemPrompts, selectedSystemPromptId]
  );

  const systemPromptOptions = useMemo<ComboboxOption[]>(
    () => (systemPrompts ?? []).map((p) => ({ label: p.name, value: p.id })),
    [systemPrompts]
  );

  const systemPromptValue = useMemo(
    () => systemPromptOptions.find((o) => o.value === selectedSystemPromptId) ?? null,
    [systemPromptOptions, selectedSystemPromptId]
  );

  // Auto-select the first prompt once the list loads if nothing is selected yet.
  useEffect(() => {
    if (selectedSystemPromptId === null && systemPrompts && systemPrompts.length > 0 && systemPrompts[0]) {
      dispatch(selectedSystemPromptIdChanged(systemPrompts[0].id));
    }
  }, [dispatch, selectedSystemPromptId, systemPrompts]);

  const handleModelChange = useCallback(
    (model: AnyModelConfig) => {
      dispatch(selectedModelKeyChanged(model.key));
    },
    [dispatch]
  );

  const handleSystemPromptChange = useCallback<ComboboxOnChange>(
    (option) => {
      dispatch(selectedSystemPromptIdChanged(option?.value ?? null));
    },
    [dispatch]
  );

  const handleManagePrompts = useCallback(() => {
    openSystemPromptsModal();
  }, []);

  const noOptionsMessage = useCallback(() => t('systemPrompts.noPromptsYet'), [t]);

  const handleExpand = useCallback(async () => {
    if (!selectedModel || !prompt.trim()) {
      return;
    }
    try {
      const result = await expandPrompt({
        prompt,
        model_key: selectedModel.key,
        system_prompt: selectedSystemPrompt?.content,
      }).unwrap();
      if (result.expanded_prompt) {
        setPromptUndo(prompt);
        dispatch(positivePromptChanged(result.expanded_prompt));
      }
      popover.close();
    } catch {
      // Error is handled by RTK Query
    }
  }, [selectedModel, prompt, expandPrompt, selectedSystemPrompt, dispatch, popover]);

  const handleOpenModelManager = useCallback(() => {
    popover.close();
    navigationApi.switchToTab('models');
    setInstallModelsTabByName('starterModels');
  }, [popover]);

  return (
    <Popover
      isOpen={popover.isOpen}
      onOpen={popover.open}
      onClose={popover.close}
      placement="left-start"
      isLazy
      closeOnBlur={false}
    >
      <PopoverTrigger>
        <span>
          <Tooltip label={hasModels ? t('prompt.expandPromptWithLLM') : t('prompt.noTextLLMInstalledTitle')}>
            <IconButton
              size="sm"
              variant="promptOverlay"
              aria-label={t('prompt.expandPromptWithLLM')}
              icon={<PiSparkleBold />}
              sx={isLoading ? loadingStyles : undefined}
              isDisabled={isLoading || (hasModels && !prompt.trim())}
            />
          </Tooltip>
        </span>
      </PopoverTrigger>
      <Portal>
        <PopoverContent p={3} w={380}>
          <PopoverArrow />
          <PopoverBody p={0}>
            {hasModels ? (
              <Flex flexDir="column" gap={3}>
                <Text fontWeight="semibold" fontSize="sm">
                  {t('prompt.expandPrompt')}
                </Text>

                <FormControl orientation="vertical">
                  <FormLabel m={0}>{t('systemPrompts.systemPrompt')}</FormLabel>
                  <Flex gap={2} alignItems="center" w="full">
                    <Flex flex={1} minW={0}>
                      <Combobox
                        value={systemPromptValue}
                        options={systemPromptOptions}
                        onChange={handleSystemPromptChange}
                        placeholder={t('systemPrompts.selectSystemPrompt')}
                        isClearable={false}
                        noOptionsMessage={noOptionsMessage}
                      />
                    </Flex>
                    <Tooltip label={t('systemPrompts.manageSystemPrompts')}>
                      <IconButton
                        aria-label={t('systemPrompts.manageSystemPrompts')}
                        icon={<PiPencilSimpleBold />}
                        size="sm"
                        variant="ghost"
                        onClick={handleManagePrompts}
                      />
                    </Tooltip>
                  </Flex>
                </FormControl>

                <ModelPicker
                  pickerId="expand-prompt-model"
                  modelConfigs={modelConfigs}
                  selectedModelConfig={selectedModel}
                  onChange={handleModelChange}
                  placeholder={t('prompt.selectTextLLM')}
                />
                <Button
                  size="sm"
                  colorScheme="invokeBlue"
                  onClick={handleExpand}
                  isLoading={isLoading}
                  isDisabled={!selectedModel || !prompt.trim()}
                >
                  {t('prompt.expand')}
                </Button>
              </Flex>
            ) : (
              <Flex flexDir="column" gap={3}>
                <Text fontWeight="semibold" fontSize="sm">
                  {t('prompt.noTextLLMInstalledTitle')}
                </Text>
                <Text fontSize="sm" color="base.300">
                  {t('prompt.noTextLLMInstalledDescription')}
                </Text>
                <Button size="sm" colorScheme="invokeBlue" onClick={handleOpenModelManager}>
                  {t('prompt.openModelManager')}
                </Button>
              </Flex>
            )}
          </PopoverBody>
        </PopoverContent>
      </Portal>
    </Popover>
  );
});

ExpandPromptButton.displayName = 'ExpandPromptButton';
