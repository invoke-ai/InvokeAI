import type { SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Button,
  Flex,
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
import { LLMTaskProgressDisplay } from 'features/prompt/LLMTaskProgressDisplay';
import { setPromptUndo } from 'features/prompt/promptUndo';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSparkleBold } from 'react-icons/pi';
import { useExpandPromptMutation } from 'services/api/endpoints/utilities';
import { useTextLLMModels } from 'services/api/hooks/modelsByType';
import type { AnyModelConfig } from 'services/api/types';
import { clearLLMTaskState } from 'services/events/stores';
import { v4 as uuidv4 } from 'uuid';

const loadingStyles: SystemStyleObject = {
  svg: { animation: spinAnimation },
};

export const ExpandPromptButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const prompt = useAppSelector(selectPositivePrompt);
  const [modelConfigs] = useTextLLMModels();
  const popover = useDisclosure(false);
  const [selectedModel, setSelectedModel] = useState<AnyModelConfig | undefined>(undefined);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [expandPrompt, { isLoading }] = useExpandPromptMutation();

  const hasModels = modelConfigs.length > 0;

  const handleModelChange = useCallback((model: AnyModelConfig) => {
    setSelectedModel(model);
  }, []);

  const handleExpand = useCallback(async () => {
    if (!selectedModel || !prompt.trim()) {
      return;
    }
    const newTaskId = uuidv4();
    setTaskId(newTaskId);
    try {
      const result = await expandPrompt({
        prompt,
        model_key: selectedModel.key,
        task_id: newTaskId,
      }).unwrap();
      if (result.expanded_prompt) {
        setPromptUndo(prompt);
        dispatch(positivePromptChanged(result.expanded_prompt));
      }
      popover.close();
    } catch {
      // Error is handled by RTK Query
    } finally {
      clearLLMTaskState(newTaskId);
      setTaskId(null);
    }
  }, [selectedModel, prompt, expandPrompt, dispatch, popover]);

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
        <PopoverContent p={3} w={350}>
          <PopoverArrow />
          <PopoverBody p={0}>
            {hasModels ? (
              <Flex flexDir="column" gap={3}>
                <Text fontWeight="semibold" fontSize="sm">
                  {t('prompt.expandPrompt')}
                </Text>
                <ModelPicker
                  pickerId="expand-prompt-model"
                  modelConfigs={modelConfigs}
                  selectedModelConfig={selectedModel}
                  onChange={handleModelChange}
                  placeholder={t('prompt.selectTextLLM')}
                />
                {isLoading ? <LLMTaskProgressDisplay taskId={taskId} /> : null}
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
