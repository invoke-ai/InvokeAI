import { Flex, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { UploadImageIconButton } from 'common/hooks/useImageUploadButton';
import { imageDTOToImageWithDims } from 'features/controlLayers/store/util';
import { DndImage } from 'features/dnd/DndImage';
import { DndImageIcon } from 'features/dnd/DndImageIcon';
import { selectStartingFrameImage, startingFrameImageChanged } from 'features/parameters/store/videoSlice';
import { t } from 'i18next';
import { useCallback } from 'react';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';
import { useImageDTO } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

export const StartingFrameImage = () => {
  const dispatch = useAppDispatch();
  const startingFrameImage = useAppSelector(selectStartingFrameImage);
  const imageDTO = useImageDTO(startingFrameImage?.image_name);
 

  const onReset = useCallback(() => {
    dispatch(startingFrameImageChanged(null));
  }, [dispatch]);

  const onUpload = useCallback(
    (imageDTO: ImageDTO) => {
      dispatch(startingFrameImageChanged(imageDTOToImageWithDims(imageDTO)));
    },
    [dispatch]
  );

  return (
    <Flex justifyContent="flex-start" flexDir="column" gap={2}>
      <Flex position="relative" w={36} h={36} alignItems="center" justifyContent="center">
        {!imageDTO && <UploadImageIconButton w="full" h="full" isError={!imageDTO} onUpload={onUpload} fontSize={36} />}
        {imageDTO && (
          <>
            <DndImage imageDTO={imageDTO} borderRadius="base" />
            <Flex position="absolute" flexDir="column" top={1} insetInlineEnd={1} gap={1}>
              <DndImageIcon
                onClick={onReset}
                icon={<PiArrowCounterClockwiseBold size={16} />}
                tooltip={t('common.reset')}
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
        
      </Flex>
    </Flex>
  );
};
