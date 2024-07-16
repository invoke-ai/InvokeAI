import { Flex, useShiftModifier } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import IAIDndImageIcon from 'common/components/IAIDndImageIcon';
import { bboxHeightChanged, bboxWidthChanged, iiReset } from 'features/controlLayers/store/canvasV2Slice';
import { selectOptimalDimension } from 'features/controlLayers/store/selectors';
import type { ImageDraggableData, InitialImageDropData } from 'features/dnd/types';
import { calculateNewSize } from 'features/parameters/components/DocumentSize/calculateNewSize';
import { memo, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiRulerBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

export const InitialImagePreview = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const initialImage = useAppSelector((s) => s.canvasV2.initialImage);
  const isConnected = useAppSelector((s) => s.system.isConnected);
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const shift = useShiftModifier();

  const { currentData: imageDTO, isError: isErrorControlImage } = useGetImageDTOQuery(
    initialImage.imageObject?.image.name ?? skipToken
  );

  const onReset = useCallback(() => {
    dispatch(iiReset());
  }, [dispatch]);

  const onUseSize = useCallback(() => {
    if (!imageDTO) {
      return;
    }

    const options = { updateAspectRatio: true, clamp: true };
    if (shift) {
      const { width, height } = imageDTO;
      dispatch(bboxWidthChanged({ width, ...options }));
      dispatch(bboxHeightChanged({ height, ...options }));
    } else {
      const { width, height } = calculateNewSize(imageDTO.width / imageDTO.height, optimalDimension * optimalDimension);
      dispatch(bboxWidthChanged({ width, ...options }));
      dispatch(bboxHeightChanged({ height, ...options }));
    }
  }, [imageDTO, dispatch, optimalDimension, shift]);

  const draggableData = useMemo<ImageDraggableData | undefined>(() => {
    if (imageDTO) {
      return {
        id: 'initial_image',
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO },
      };
    }
  }, [imageDTO]);

  const droppableData = useMemo<InitialImageDropData>(
    () => ({ id: 'initial_image', actionType: 'SET_INITIAL_IMAGE' }),
    []
  );

  useEffect(() => {
    if (isConnected && isErrorControlImage) {
      onReset();
    }
  }, [onReset, isConnected, isErrorControlImage]);

  return (
    <Flex w="full" alignItems="center" justifyContent="center">
      <Flex position="relative" w="full" h="full" alignItems="center" justifyContent="center">
        <IAIDndImage
          draggableData={draggableData}
          droppableData={droppableData}
          imageDTO={imageDTO}
          // postUploadAction={postUploadAction}
        />

        {imageDTO && (
          <Flex position="absolute" flexDir="column" top={1} insetInlineEnd={1} gap={1}>
            <IAIDndImageIcon
              onClick={onReset}
              icon={<PiArrowCounterClockwiseBold size={16} />}
              tooltip={t('controlnet.resetControlImage')}
            />
            <IAIDndImageIcon
              onClick={onUseSize}
              icon={<PiRulerBold size={16} />}
              tooltip={
                shift ? t('controlnet.setControlImageDimensionsForce') : t('controlnet.setControlImageDimensions')
              }
            />
          </Flex>
        )}
      </Flex>
    </Flex>
  );
});

InitialImagePreview.displayName = 'InitialImagePreview';
