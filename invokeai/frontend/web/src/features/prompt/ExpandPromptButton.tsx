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
import { ModelPicker } from 'features/parameters/components/ModelPicker';
import { setPromptUndo } from 'features/prompt/promptUndo';
import { memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSparkleBold } from 'react-icons/pi';
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
  const [modelConfigs] = useTextLLMModels();
  const popover = useDisclosure(false);
  const [selectedModel, setSelectedModel] = useState<AnyModelConfig | undefined>(undefined);
  const [expandPrompt, { isLoading }] = useExpandPromptMutation();

  const handleModelChange = useCallback((model: AnyModelConfig) => {
    setSelectedModel(model);
  }, []);

  const handleExpand = useCallback(async () => {
    if (!selectedModel || !prompt.trim()) {
      return;
    }
    try {
      const result = await expandPrompt({
        prompt,
        model_key: selectedModel.key,
      }).unwrap();
      if (result.expanded_prompt) {
        setPromptUndo(prompt);
        dispatch(positivePromptChanged(result.expanded_prompt));
      }
      popover.close();
    } catch {
      // Error is handled by RTK Query
    }
  }, [selectedModel, prompt, expandPrompt, dispatch, popover]);

  // Don't render if no TextLLM models are installed
  if (modelConfigs.length === 0) {
    return null;
  }

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
          <Tooltip label={t('prompt.expandPromptWithLLM')}>
            <IconButton
              size="sm"
              variant="promptOverlay"
              aria-label={t('prompt.expandPromptWithLLM')}
              icon={<PiSparkleBold />}
              sx={isLoading ? loadingStyles : undefined}
              isDisabled={isLoading || !prompt.trim()}
            />
          </Tooltip>
        </span>
      </PopoverTrigger>
      <Portal>
        <PopoverContent p={3} w={350}>
          <PopoverArrow />
          <PopoverBody p={0}>
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
          </PopoverBody>
        </PopoverContent>
      </Portal>
    </Popover>
  );
});

ExpandPromptButton.displayName = 'ExpandPromptButton';
