import { Box, Button, Flex, Grid, Heading, Icon, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { setUpscaleInitialImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { DndImage } from 'features/dnd/DndImage';
import { DndImageIcon } from 'features/dnd/DndImageIcon';
import { selectUpscaleInitialImage, upscaleInitialImageChanged, creativityChanged, structureChanged, selectCreativity, selectStructure } from 'features/parameters/store/upscaleSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiImageBold, PiUploadBold, PiShieldCheckBold, PiScalesBold, PiPaletteBold, PiSparkleBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

import { LaunchpadButton } from './LaunchpadButton';

export const UpscalingLaunchpadPanel = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const upscaleInitialImage = useAppSelector(selectUpscaleInitialImage);
  const creativity = useAppSelector(selectCreativity);
  const structure = useAppSelector(selectStructure);

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

  // Preset button handlers
  const onConservativeClick = useCallback(() => {
    dispatch(creativityChanged(-5));
    dispatch(structureChanged(5));
  }, [dispatch]);

  const onBalancedClick = useCallback(() => {
    dispatch(creativityChanged(0));
    dispatch(structureChanged(0));
  }, [dispatch]);

  const onCreativeClick = useCallback(() => {
    dispatch(creativityChanged(5));
    dispatch(structureChanged(-2));
  }, [dispatch]);

  const onArtisticClick = useCallback(() => {
    dispatch(creativityChanged(8));
    dispatch(structureChanged(-5));
  }, [dispatch]);

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
              <Icon as={PiImageBold} boxSize={8} color="base.500" />
              <Flex flexDir="column" alignItems="flex-start" gap={2}>
                <Heading size="sm">{t('ui.launchpad.upscaling.replaceImage.title')}</Heading>
                <Text color="base.300">{t('ui.launchpad.upscaling.replaceImage.description')}</Text>
              </Flex>
              <Flex position="absolute" right={3} bottom={3}>
                <PiUploadBold />
                <input {...uploadApi.getUploadInputProps()} />
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
          {/* Left Column: Creativity and Structural Defaults */}
          <Box>
            <Text fontWeight="semibold" fontSize="sm" mb={3}>
              Creativity & Structural Defaults
            </Text>
            <Flex flexDir="column" gap={2}>
              <Button 
                size="sm" 
                variant={creativity === -5 && structure === 5 ? "solid" : "outline"} 
                colorScheme="base" 
                justifyContent="center"
                onClick={onConservativeClick}
                leftIcon={<PiShieldCheckBold />}
              >
                Conservative
              </Button>
              <Button 
                size="sm" 
                variant={creativity === 0 && structure === 0 ? "solid" : "outline"} 
                colorScheme="base" 
                justifyContent="center"
                onClick={onBalancedClick}
                leftIcon={<PiScalesBold />}
              >
                Balanced
              </Button>
              <Button 
                size="sm" 
                variant={creativity === 5 && structure === -2 ? "solid" : "outline"} 
                colorScheme="base" 
                justifyContent="center"
                onClick={onCreativeClick}
                leftIcon={<PiPaletteBold />}
              >
                Creative
              </Button>
              <Button 
                size="sm" 
                variant={creativity === 8 && structure === -5 ? "solid" : "outline"} 
                colorScheme="base" 
                justifyContent="center"
                onClick={onArtisticClick}
                leftIcon={<PiSparkleBold />}
              >
                Artistic
              </Button>
            </Flex>
          </Box>
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
