import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, useShiftModifier } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import IAIDndImageIcon from 'common/components/IAIDndImageIcon';
import { setBoundingBoxDimensions } from 'features/canvas/store/canvasSlice';
import { heightChanged, widthChanged } from 'features/controlLayers/store/controlLayersSlice';
import type { ImageWithDims } from 'features/controlLayers/util/controlAdapters';
import type { ControlLayerDropData, ImageDraggableData } from 'features/dnd/types';
import { calculateNewSize } from 'features/parameters/components/ImageSize/calculateNewSize';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiRulerBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { ControlLayerAction, ImageDTO } from 'services/api/types';

type Props = {
  image: ImageWithDims | null;
  onChangeImage: (imageDTO: ImageDTO | null) => void;
  layerId: string; // required for the dnd/upload interactions
};

export const IPAdapterImagePreview = memo(({ image, onChangeImage, layerId }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isConnected = useAppSelector((s) => s.system.isConnected);
  const activeTabName = useAppSelector(activeTabNameSelector);
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const shift = useShiftModifier();

  const { currentData: controlImage, isError: isErrorControlImage } = useGetImageDTOQuery(
    image?.imageName ?? skipToken
  );
  const handleResetControlImage = useCallback(() => {
    onChangeImage(null);
  }, [onChangeImage]);

  const handleSetControlImageToDimensions = useCallback(() => {
    if (!controlImage) {
      return;
    }

    if (activeTabName === 'unifiedCanvas') {
      dispatch(setBoundingBoxDimensions({ width: controlImage.width, height: controlImage.height }, optimalDimension));
    } else {
      if (shift) {
        const { width, height } = controlImage;
        dispatch(widthChanged({ width, updateAspectRatio: true }));
        dispatch(heightChanged({ height, updateAspectRatio: true }));
      } else {
        const { width, height } = calculateNewSize(
          controlImage.width / controlImage.height,
          optimalDimension * optimalDimension
        );
        dispatch(widthChanged({ width, updateAspectRatio: true }));
        dispatch(heightChanged({ height, updateAspectRatio: true }));
      }
    }
  }, [controlImage, activeTabName, dispatch, optimalDimension, shift]);

  const draggableData = useMemo<ImageDraggableData | undefined>(() => {
    if (controlImage) {
      return {
        id: layerId,
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO: controlImage },
      };
    }
  }, [controlImage, layerId]);

  const droppableData = useMemo<ControlLayerDropData>(
    () => ({
      id: layerId,
      actionType: 'SET_CONTROL_LAYER_IMAGE',
      context: { layerId },
    }),
    [layerId]
  );

  const postUploadAction = useMemo<ControlLayerAction>(() => ({ type: 'SET_CONTROL_LAYER_IMAGE', layerId }), [layerId]);

  useEffect(() => {
    if (isConnected && isErrorControlImage) {
      handleResetControlImage();
    }
  }, [handleResetControlImage, isConnected, isErrorControlImage]);

  return (
    <Flex position="relative" w="full" h={36} alignItems="center" justifyContent="center">
      <IAIDndImage
        draggableData={draggableData}
        droppableData={droppableData}
        imageDTO={controlImage}
        postUploadAction={postUploadAction}
      />

      <>
        <IAIDndImageIcon
          onClick={handleResetControlImage}
          icon={controlImage ? <PiArrowCounterClockwiseBold size={16} /> : undefined}
          tooltip={t('controlnet.resetControlImage')}
        />
        <IAIDndImageIcon
          onClick={handleSetControlImageToDimensions}
          icon={controlImage ? <PiRulerBold size={16} /> : undefined}
          tooltip={shift ? t('controlnet.setControlImageDimensionsForce') : t('controlnet.setControlImageDimensions')}
          styleOverrides={setControlImageDimensionsStyleOverrides}
        />
      </>
    </Flex>
  );
});

IPAdapterImagePreview.displayName = 'IPAdapterImagePreview';

const setControlImageDimensionsStyleOverrides: SystemStyleObject = { mt: 6 };
