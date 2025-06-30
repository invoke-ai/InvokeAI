import { Box, Flex, Grid, Heading, Icon, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { setUpscaleInitialImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { DndImage } from 'features/dnd/DndImage';
import { DndImageIcon } from 'features/dnd/DndImageIcon';
import ParamSpandrelModel from 'features/parameters/components/Upscale/ParamSpandrelModel';
import { selectUpscaleInitialImage, upscaleInitialImageChanged } from 'features/parameters/store/upscaleSlice';
import { MainModelPicker } from 'features/settingsAccordions/components/GenerationSettingsAccordion/MainModelPicker';
import { UpscaleScaleSlider } from 'features/settingsAccordions/components/UpscaleSettingsAccordion/UpscaleScaleSlider';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiImageBold, PiUploadBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

import { LaunchpadButton } from './LaunchpadButton';

export const UpscalingLaunchpadPanel = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const upscaleInitialImage = useAppSelector(selectUpscaleInitialImage);

  const dndTargetData = useMemo(() => setUpscaleInitialImageDndTarget.getData(), []);

  const onUpload = useCallback(
    (imageDTO: ImageDTO) => {
      dispatch(upscaleInitialImageChanged(imageDTO));
    },
    [dispatch]
  );

  const onReset = useCallback(() => {
    dispatch(upscaleInitialImageChanged(null));
  }, [dispatch]);

  const uploadApi = useImageUploadButton({ allowMultiple: false, onUpload });

  return (
    <Flex flexDir="column" h="full" w="full" alignItems="center" gap={2}>
      <Flex flexDir="column" w="full" gap={4} px={14} maxW={768} pt="20vh">
        <Heading mb={4}>{t('ui.launchpad.upscalingTitle')}</Heading>

        {/* Upload Area */}
        <LaunchpadButton {...uploadApi.getUploadButtonProps()} position="relative" gap={8}>
          {!upscaleInitialImage ? (
            <>
              <Icon as={PiImageBold} boxSize={8} color="base.500" />
              <Flex flexDir="column" alignItems="flex-start" gap={2}>
                <Heading size="sm">{t('ui.launchpad.upscaling.uploadImage.title')}</Heading>
                <Text color="base.300">{t('ui.launchpad.upscaling.uploadImage.description')}</Text>
              </Flex>
              <Flex position="absolute" right={3} bottom={3}>
                <PiUploadBold />
                <input {...uploadApi.getUploadInputProps()} />
              </Flex>
            </>
          ) : (
            <>
              <Flex position="relative" w={16} h={16} alignItems="center" justifyContent="center">
                <DndImage imageDTO={upscaleInitialImage} />
                <Flex position="absolute" flexDir="column" top={1} insetInlineEnd={1} gap={1}>
                  <DndImageIcon
                    onClick={onReset}
                    icon={<PiArrowCounterClockwiseBold size={12} />}
                    tooltip={t('common.reset')}
                  />
                </Flex>
                <Text
                  position="absolute"
                  background="base.900"
                  color="base.50"
                  fontSize="xs"
                  fontWeight="semibold"
                  bottom={0}
                  left={0}
                  opacity={0.7}
                  px={1}
                  lineHeight={1.25}
                  borderTopEndRadius="base"
                  borderBottomStartRadius="base"
                  pointerEvents="none"
                >{`${upscaleInitialImage.width}x${upscaleInitialImage.height}`}</Text>
              </Flex>
              <Flex flexDir="column" alignItems="flex-start" gap={2}>
                <Heading size="sm">{t('ui.launchpad.upscaling.imageReady.title')}</Heading>
                <Text color="base.300">{t('ui.launchpad.upscaling.imageReady.description')}</Text>
              </Flex>
            </>
          )}
          <DndDropTarget
            dndTarget={setUpscaleInitialImageDndTarget}
            dndTargetData={dndTargetData}
            label={t('gallery.drop')}
          />
        </LaunchpadButton>

        {/* Guidance text */}
        {upscaleInitialImage && (
          <Flex bg="base.800" p={4} borderRadius="base" border="1px solid" borderColor="base.700" mt={6}>
            <Text variant="subtext" fontSize="sm" lineHeight="1.6">
              <strong>{t('ui.launchpad.upscaling.readyToUpscale.title')}</strong>{' '}
              {t('ui.launchpad.upscaling.readyToUpscale.description')}
            </Text>
          </Flex>
        )}

        {/* Controls */}
        <style>{`.launchpad-hide-label .chakra-form__label { display: none !important; }`}</style>
        <Grid gridTemplateColumns="1fr 1fr" gap={10} alignItems="start" mt={upscaleInitialImage ? 8 : 12}>
          {/* Left Column: All parameters stacked */}
          <Flex flexDir="column" gap={6} alignItems="flex-start">
            <Box w="full">
              <Text fontWeight="semibold" fontSize="sm" mb={1}>
                {t('ui.launchpad.upscaling.upscaleModel')}
              </Text>
              <Box className="launchpad-hide-label">
                <ParamSpandrelModel />
              </Box>
            </Box>
            <Box w="full">
              <Text fontWeight="semibold" fontSize="sm" mb={1}>
                {t('ui.launchpad.upscaling.model')}
              </Text>
              <Box className="launchpad-hide-label">
                <MainModelPicker />
              </Box>
            </Box>
            <Box w="full">
              <Text fontWeight="semibold" fontSize="sm" mb={1}>
                {t('ui.launchpad.upscaling.scale')}
              </Text>
              <Box className="launchpad-hide-label">
                <UpscaleScaleSlider />
              </Box>
            </Box>
          </Flex>
          {/* Right Column: Description/help text */}
          <Box>
            <Text variant="subtext" fontSize="sm" lineHeight="1.6">
              {t('ui.launchpad.upscaling.helpText.promptAdvice')}
            </Text>
            <Text variant="subtext" fontSize="sm" lineHeight="1.6" mt={3}>
              {t('ui.launchpad.upscaling.helpText.styleAdvice')}
            </Text>
          </Box>
        </Grid>
      </Flex>
    </Flex>
  );
});

UpscalingLaunchpadPanel.displayName = 'UpscalingLaunchpadPanel';
