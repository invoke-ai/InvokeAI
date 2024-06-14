import { Flex, useShiftModifier } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import IAIDndImageIcon from 'common/components/IAIDndImageIcon';
import { heightChanged, widthChanged } from 'features/controlLayers/store/controlLayersSlice';
import type { ImageWithDims } from 'features/controlLayers/store/types';
import type { ImageDraggableData, TypesafeDroppableData } from 'features/dnd/types';
import { calculateNewSize } from 'features/parameters/components/ImageSize/calculateNewSize';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { memo, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiRulerBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { ImageDTO, PostUploadAction } from 'services/api/types';

type Props = {
  image: ImageWithDims | null;
  onChangeImage: (imageDTO: ImageDTO | null) => void;
  ipAdapterId: string; // required for the dnd/upload interactions
  droppableData: TypesafeDroppableData;
  postUploadAction: PostUploadAction;
};

export const IPAImagePreview = memo(({ image, onChangeImage, ipAdapterId, droppableData, postUploadAction }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isConnected = useAppSelector((s) => s.system.isConnected);
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const shift = useShiftModifier();

  const { currentData: controlImage, isError: isErrorControlImage } = useGetImageDTOQuery(image?.name ?? skipToken);
  const handleResetControlImage = useCallback(() => {
    onChangeImage(null);
  }, [onChangeImage]);

  const handleSetControlImageToDimensions = useCallback(() => {
    if (!controlImage) {
      return;
    }

    const options = { updateAspectRatio: true, clamp: true };
    if (shift) {
      const { width, height } = controlImage;
      dispatch(widthChanged({ width, ...options }));
      dispatch(heightChanged({ height, ...options }));
    } else {
      const { width, height } = calculateNewSize(
        controlImage.width / controlImage.height,
        optimalDimension * optimalDimension
      );
      dispatch(widthChanged({ width, ...options }));
      dispatch(heightChanged({ height, ...options }));
    }
  }, [controlImage, dispatch, optimalDimension, shift]);

  const draggableData = useMemo<ImageDraggableData | undefined>(() => {
    if (controlImage) {
      return {
        id: ipAdapterId,
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO: controlImage },
      };
    }
  }, [controlImage, ipAdapterId]);

  useEffect(() => {
    if (isConnected && isErrorControlImage) {
      handleResetControlImage();
    }
  }, [handleResetControlImage, isConnected, isErrorControlImage]);

  return (
    <Flex position="relative" w={36} h={36} alignItems="center">
      <IAIDndImage
        draggableData={draggableData}
        droppableData={droppableData}
        imageDTO={controlImage}
        postUploadAction={postUploadAction}
      />

      {controlImage && (
        <Flex position="absolute" flexDir="column" top={1} insetInlineEnd={1} gap={1}>
          <IAIDndImageIcon
            onClick={handleResetControlImage}
            icon={<PiArrowCounterClockwiseBold size={16} />}
            tooltip={t('controlnet.resetControlImage')}
          />
          <IAIDndImageIcon
            onClick={handleSetControlImageToDimensions}
            icon={<PiRulerBold size={16} />}
            tooltip={shift ? t('controlnet.setControlImageDimensionsForce') : t('controlnet.setControlImageDimensions')}
          />
        </Flex>
      )}
    </Flex>
  );
});

IPAImagePreview.displayName = 'IPAImagePreview';
