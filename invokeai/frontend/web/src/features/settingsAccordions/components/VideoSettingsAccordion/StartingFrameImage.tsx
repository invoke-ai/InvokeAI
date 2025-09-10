import { Flex, FormLabel, Icon, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { UploadImageIconButton } from 'common/hooks/useImageUploadButton';
import { ASPECT_RATIO_MAP } from 'features/controlLayers/store/types';
import { imageDTOToImageWithDims } from 'features/controlLayers/store/util';
import { videoFrameFromImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { DndImage } from 'features/dnd/DndImage';
import { DndImageIcon } from 'features/dnd/DndImageIcon';
import {
  selectStartingFrameImage,
  selectVideoAspectRatio,
  selectVideoModelRequiresStartingFrame,
  startingFrameImageChanged,
} from 'features/parameters/store/videoSlice';
import { t } from 'i18next';
import { useCallback, useMemo } from 'react';
import { PiArrowCounterClockwiseBold, PiWarningBold } from 'react-icons/pi';
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

  const fitsCurrentAspectRatio = useMemo(() => {
    if (!imageDTO) {
      return true;
    }
    console.log('imageDTO.width / imageDTO.height', imageDTO.width / imageDTO.height);
    console.log('ASPECT_RATIO_MAP[videoAspectRatio]?.ratio', ASPECT_RATIO_MAP[videoAspectRatio]?.ratio);
    console.log('fitsCurrentAspectRatio', imageDTO.width / imageDTO.height === ASPECT_RATIO_MAP[videoAspectRatio]?.ratio);
    return imageDTO.width / imageDTO.height === ASPECT_RATIO_MAP[videoAspectRatio]?.ratio;
  }, [imageDTO, videoAspectRatio]);

  return (
    <Flex justifyContent="flex-start" flexDir="column" gap={2}>
      <FormLabel>{t('parameters.startingFrameImage')}</FormLabel>
      <Flex position="relative" w={36} h={36} alignItems="center" justifyContent="center" borderWidth={1} borderStyle="solid" borderColor={fitsCurrentAspectRatio ? 'base.500' : 'warning.500'}>
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

            {!fitsCurrentAspectRatio && <Tooltip
              label={t('parameters.startingFrameImageAspectRatioWarning', { videoAspectRatio: videoAspectRatio })}
            >
              <Flex position="absolute" flexDir="column" top={1} insetInlineStart={1} gap={1}>
                <Icon
                  as={PiWarningBold}
                  size={16}
                  color="warning.500"
                />
              </Flex>
            </Tooltip>}

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
