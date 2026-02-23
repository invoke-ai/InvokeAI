import type { SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Button,
  Flex,
  IconButton,
  Image,
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
import { useAppDispatch } from 'app/store/storeHooks';
import { useDisclosure } from 'common/hooks/useBoolean';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { positivePromptChanged } from 'features/controlLayers/store/paramsSlice';
import { ModelPicker } from 'features/parameters/components/ModelPicker';
import { memo, useCallback, useState } from 'react';
import { PiImageBold } from 'react-icons/pi';
import { useImageToPromptMutation } from 'services/api/endpoints/utilities';
import { useLlavaModels } from 'services/api/hooks/modelsByType';
import type { AnyModelConfig, ImageDTO } from 'services/api/types';

const loadingStyles: SystemStyleObject = {
  svg: { animation: spinAnimation },
};

export const ImageToPromptButton = memo(() => {
  const dispatch = useAppDispatch();
  const [modelConfigs] = useLlavaModels();
  const popover = useDisclosure(false);
  const [selectedModel, setSelectedModel] = useState<AnyModelConfig | undefined>(undefined);
  const [uploadedImage, setUploadedImage] = useState<ImageDTO | undefined>(undefined);
  const [imageToPrompt, { isLoading }] = useImageToPromptMutation();

  const handleModelChange = useCallback((model: AnyModelConfig) => {
    setSelectedModel(model);
  }, []);

  const handleImageUpload = useCallback((imageDTO: ImageDTO) => {
    setUploadedImage(imageDTO);
  }, []);

  const { getUploadButtonProps, getUploadInputProps } = useImageUploadButton({
    allowMultiple: false,
    onUpload: handleImageUpload,
  });

  const handleGenerate = useCallback(async () => {
    if (!selectedModel || !uploadedImage) {
      return;
    }
    try {
      const result = await imageToPrompt({
        image_name: uploadedImage.image_name,
        model_key: selectedModel.key,
      }).unwrap();
      if (result.prompt) {
        dispatch(positivePromptChanged(result.prompt));
      }
      popover.close();
      setUploadedImage(undefined);
    } catch {
      // Error is handled by RTK Query
    }
  }, [selectedModel, uploadedImage, imageToPrompt, dispatch, popover]);

  const handleClose = useCallback(() => {
    popover.close();
    setUploadedImage(undefined);
  }, [popover]);

  // Don't render if no LLaVA models are installed
  if (modelConfigs.length === 0) {
    return null;
  }

  return (
    <Popover
      isOpen={popover.isOpen}
      onOpen={popover.open}
      onClose={handleClose}
      placement="left-start"
      isLazy
      closeOnBlur={false}
    >
      <PopoverTrigger>
        <span>
          <Tooltip label="Image to Prompt">
            <IconButton
              size="sm"
              variant="promptOverlay"
              aria-label="Image to Prompt"
              icon={<PiImageBold />}
              sx={isLoading ? loadingStyles : undefined}
              isDisabled={isLoading}
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
                Image to Prompt
              </Text>
              <ModelPicker
                pickerId="image-to-prompt-model"
                modelConfigs={modelConfigs}
                selectedModelConfig={selectedModel}
                onChange={handleModelChange}
                placeholder="Select Vision Model..."
              />
              <Flex gap={2} alignItems="center">
                <Button size="sm" variant="outline" flexGrow={1} {...getUploadButtonProps()}>
                  {uploadedImage ? 'Change Image' : 'Upload Image'}
                </Button>
                <input {...getUploadInputProps()} />
                {uploadedImage && (
                  <Image
                    src={uploadedImage.image_url}
                    alt="Uploaded"
                    boxSize={10}
                    objectFit="cover"
                    borderRadius="base"
                  />
                )}
              </Flex>
              <Button
                size="sm"
                colorScheme="invokeBlue"
                onClick={handleGenerate}
                isLoading={isLoading}
                isDisabled={!selectedModel || !uploadedImage}
              >
                Generate Prompt
              </Button>
            </Flex>
          </PopoverBody>
        </PopoverContent>
      </Portal>
    </Popover>
  );
});

ImageToPromptButton.displayName = 'ImageToPromptButton';
