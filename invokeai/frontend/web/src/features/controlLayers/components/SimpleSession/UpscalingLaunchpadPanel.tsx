import { Box, Flex, FormControl, FormLabel, Heading, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import ParamSpandrelModel from 'features/parameters/components/Upscale/ParamSpandrelModel';
import { selectUpscaleInitialImage } from 'features/parameters/store/upscaleSlice';
import { MainModelPicker } from 'features/settingsAccordions/components/GenerationSettingsAccordion/MainModelPicker';
import { UpscaleScaleSlider } from 'features/settingsAccordions/components/UpscaleSettingsAccordion/UpscaleScaleSlider';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImageBold } from 'react-icons/pi';

import { LaunchpadButton } from './LaunchpadButton';

export const UpscalingLaunchpadPanel = memo(() => {
  const { t } = useTranslation();
  const upscaleInitialImage = useAppSelector(selectUpscaleInitialImage);

  const onUpload = useCallback(() => {
    // Upload handler will be called automatically by the upload hook
  }, []);

  const uploadApi = useImageUploadButton({ allowMultiple: false, onUpload });

  return (
    <Flex flexDir="column" h="full" w="full" alignItems="center" gap={2}>
      <Flex flexDir="column" w="full" gap={6} px={14} maxW={768} pt="20vh">
        <Heading mb={4}>{t('upscaling.upscale')} and add detail.</Heading>

        {/* Upload Area - First CTA as per Devon's feedback */}
        <LaunchpadButton {...uploadApi.getUploadButtonProps()} position="relative" gap={8} h={24}>
          <Flex flexDir="column" alignItems="center" gap={2}>
            <PiImageBold size={32} />
            <Box textAlign="center">
              {!upscaleInitialImage && (
                <>
                  <Text fontWeight="semibold">Click or drag an image to upscale</Text>
                  <Text variant="subtext" fontSize="sm">
                    JPG, PNG, WebP up to 100MB
                  </Text>
                </>
              )}
              {upscaleInitialImage && (
                <>
                  <Text fontWeight="semibold">Image ready</Text>
                  <Text variant="subtext" fontSize="sm">
                    Press Invoke to begin upscaling
                  </Text>
                </>
              )}
            </Box>
          </Flex>
        </LaunchpadButton>

        {/* Controls - 60% width as per Devon's feedback */}
        <Flex flexDir="column" gap={4} w="60%">
          {/* Upscale Model */}
          <FormControl>
            <FormLabel>{t('upscaling.upscaleModel')}</FormLabel>
            <ParamSpandrelModel />
          </FormControl>

          {/* Generation Model */}
          <FormControl>
            <FormLabel>{t('parameters.model')}</FormLabel>
            <MainModelPicker />
          </FormControl>

          {/* Scale Slider - Same width as dropdowns */}
          <UpscaleScaleSlider />
        </Flex>

        {/* Description text with paragraph breaks as per Devon's feedback */}
        <Box w="full" maxW="40%" ml="auto">
          <Text variant="subtext" fontSize="sm" lineHeight="1.6">
            When upscaling, use a prompt that describes the medium and style. Avoid describing specific content details in the image.
          </Text>
          <Text variant="subtext" fontSize="sm" lineHeight="1.6" mt={3}>
            Upscaling works best with the general style of your image.
          </Text>
        </Box>
      </Flex>
    </Flex>
  );
});

UpscalingLaunchpadPanel.displayName = 'UpscalingLaunchpadPanel';
