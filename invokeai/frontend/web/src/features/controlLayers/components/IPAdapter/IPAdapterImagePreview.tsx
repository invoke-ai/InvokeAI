import { Flex, useShiftModifier } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import IAIDndImageIcon from 'common/components/IAIDndImageIcon';
import { useNanoid } from 'common/hooks/useNanoid';
import { bboxHeightChanged, bboxWidthChanged } from 'features/controlLayers/store/canvasSlice';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { selectOptimalDimension } from 'features/controlLayers/store/selectors';
import type { ImageWithDims } from 'features/controlLayers/store/types';
import type { ImageDraggableData, TypesafeDroppableData } from 'features/dnd/types';
import { calculateNewSize } from 'features/parameters/components/Bbox/calculateNewSize';
import { memo, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiRulerBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { ImageDTO, PostUploadAction } from 'services/api/types';
import { $isConnected } from 'services/events/stores';

type Props = {
  image: ImageWithDims | null;
  onChangeImage: (imageDTO: ImageDTO | null) => void;
  droppableData: TypesafeDroppableData;
  postUploadAction: PostUploadAction;
};

export const IPAdapterImagePreview = memo(({ image, onChangeImage, droppableData, postUploadAction }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isStaging = useAppSelector(selectIsStaging);
  const isConnected = useStore($isConnected);
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const shift = useShiftModifier();
  const dndId = useNanoid('ip_adapter_image_preview');

  const { currentData: controlImage, isError: isErrorControlImage } = useGetImageDTOQuery(
    image?.image_name ?? skipToken
  );
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
      dispatch(bboxWidthChanged({ width, ...options }));
      dispatch(bboxHeightChanged({ height, ...options }));
    } else {
      const { width, height } = calculateNewSize(
        controlImage.width / controlImage.height,
        optimalDimension * optimalDimension
      );
      dispatch(bboxWidthChanged({ width, ...options }));
      dispatch(bboxHeightChanged({ height, ...options }));
    }
  }, [controlImage, dispatch, optimalDimension, shift]);

  const draggableData = useMemo<ImageDraggableData | undefined>(() => {
    if (controlImage) {
      return {
        id: dndId,
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO: controlImage },
      };
    }
  }, [controlImage, dndId]);

  useEffect(() => {
    if (isConnected && isErrorControlImage) {
      handleResetControlImage();
    }
  }, [handleResetControlImage, isConnected, isErrorControlImage]);

  return (
    <Flex position="relative" w="full" h="full" alignItems="center">
      <IAIDndImage
        draggableData={draggableData}
        droppableData={droppableData}
        imageDTO={controlImage}
        postUploadAction={postUploadAction}
      />

      {controlImage && (
        <Flex position="absolute" flexDir="column" top={2} insetInlineEnd={2} gap={1}>
          <IAIDndImageIcon
            onClick={handleResetControlImage}
            icon={<PiArrowCounterClockwiseBold size={16} />}
            tooltip={t('common.reset')}
          />
          <IAIDndImageIcon
            onClick={handleSetControlImageToDimensions}
            icon={<PiRulerBold size={16} />}
            tooltip={shift ? t('controlLayers.useSizeIgnoreModel') : t('controlLayers.useSizeOptimizeForModel')}
            isDisabled={isStaging}
          />
        </Flex>
      )}
    </Flex>
  );
});

IPAdapterImagePreview.displayName = 'IPAdapterImagePreview';
