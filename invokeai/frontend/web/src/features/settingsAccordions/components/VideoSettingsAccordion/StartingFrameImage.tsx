import { Box, Flex, FormLabel, Icon, IconButton, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { UploadImageIconButton } from 'common/hooks/useImageUploadButton';
import { ASPECT_RATIO_MAP } from 'features/controlLayers/store/types';
import { imageDTOToImageWithDims } from 'features/controlLayers/store/util';
import { videoFrameFromImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { DndImage } from 'features/dnd/DndImage';
import { DndImageIcon, imageButtonSx } from 'features/dnd/DndImageIcon';
import { openEditImageModal } from 'features/editImageModal/store';
import {
  selectStartingFrameImage,
  selectVideoAspectRatio,
  selectVideoModelRequiresStartingFrame,
  startingFrameImageChanged,
} from 'features/parameters/store/videoSlice';
import { t } from 'i18next';
import { useCallback, useMemo } from 'react';
import { PiArrowCounterClockwiseBold, PiCropBold, PiWarningBold } from 'react-icons/pi';
import { useImageDTO } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

const dndTargetData = videoFrameFromImageDndTarget.getData({ frame: 'start' });

export const StartingFrameImage = () => {
  const dispatch = useAppDispatch();
  const requiresStartingFrame = useAppSelector(selectVideoModelRequiresStartingFrame);
  const startingFrameImage = useAppSelector(selectStartingFrameImage);
  const imageDTO = useImageDTO(startingFrameImage?.image_name);
  const videoAspectRatio = useAppSelector(selectVideoAspectRatio);

  const onReset = useCallback(() => {
    dispatch(startingFrameImageChanged(null));
  }, [dispatch]);

  const onUpload = useCallback(
    (imageDTO: ImageDTO) => {
      dispatch(startingFrameImageChanged(imageDTOToImageWithDims(imageDTO)));
    },
    [dispatch]
  );

  const onOpenEditImageModal = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    openEditImageModal(imageDTO.image_name);
  }, [imageDTO]);

  const fitsCurrentAspectRatio = useMemo(() => {
    if (!imageDTO) {
      return true;
    }

    return imageDTO.width / imageDTO.height === ASPECT_RATIO_MAP[videoAspectRatio]?.ratio;
  }, [imageDTO, videoAspectRatio]);

  return (
    <Flex justifyContent="flex-start" flexDir="column" gap={2}>
      <FormLabel display="flex" alignItems="center" gap={2}>
        <Text>{t('parameters.startingFrameImage')}</Text>
        <Tooltip label={t('parameters.startingFrameImageAspectRatioWarning', { videoAspectRatio: videoAspectRatio })}>
          <Box>
            <Icon as={PiWarningBold} size={16} color="warning.500" />
          </Box>
        </Tooltip>
      </FormLabel>
      <Flex
        position="relative"
        w={36}
        h={36}
        alignItems="center"
        justifyContent="center"
        borderWidth={1}
        borderStyle="solid"
        borderColor={fitsCurrentAspectRatio ? 'base.500' : 'warning.500'}
      >
        {!imageDTO && (
          <UploadImageIconButton
            w="full"
            h="full"
            isError={requiresStartingFrame && !imageDTO}
            onUpload={onUpload}
            fontSize={36}
          />
        )}
        {imageDTO && (
          <>
            <DndImage imageDTO={imageDTO} borderRadius="base" borderWidth={1} borderStyle="solid" />
            <Flex position="absolute" flexDir="column" top={1} insetInlineEnd={1} gap={1}>
              <DndImageIcon
                onClick={onReset}
                icon={<PiArrowCounterClockwiseBold size={16} />}
                tooltip={t('common.reset')}
              />
            </Flex>

            <Flex position="absolute" flexDir="column" top={1} insetInlineStart={1} gap={1}>
              <IconButton
                variant="link"
                sx={imageButtonSx}
                aria-label={t('common.crop')}
                onClick={onOpenEditImageModal}
                icon={<PiCropBold size={16} />}
                tooltip={t('common.crop')}
              />
            </Flex>

            <Text
              position="absolute"
              background="base.900"
              color="base.50"
              fontSize="sm"
              fontWeight="semibold"
              bottom={0}
              left={0}
              opacity={0.7}
              px={2}
              lineHeight={1.25}
              borderTopEndRadius="base"
              borderBottomStartRadius="base"
              pointerEvents="none"
            >{`${imageDTO.width}x${imageDTO.height}`}</Text>
          </>
        )}
        <DndDropTarget label="Drop" dndTarget={videoFrameFromImageDndTarget} dndTargetData={dndTargetData} />
      </Flex>
    </Flex>
  );
};
