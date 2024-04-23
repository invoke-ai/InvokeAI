import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { aspectRatioChanged, setBoundingBoxDimensions } from 'features/canvas/store/canvasSlice';
import ParamBoundingBoxHeight from 'features/parameters/components/Canvas/BoundingBox/ParamBoundingBoxHeight';
import ParamBoundingBoxWidth from 'features/parameters/components/Canvas/BoundingBox/ParamBoundingBoxWidth';
import { AspectRatioIconPreview } from 'features/parameters/components/ImageSize/AspectRatioIconPreview';
import { ImageSize } from 'features/parameters/components/ImageSize/ImageSize';
import type { AspectRatioState } from 'features/parameters/components/ImageSize/types';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';

export const ImageSizeCanvas = memo(() => {
  const dispatch = useAppDispatch();
  const { width, height } = useAppSelector((s) => s.canvas.boundingBoxDimensions);
  const aspectRatioState = useAppSelector((s) => s.canvas.aspectRatio);
  const optimalDimension = useAppSelector(selectOptimalDimension);

  const onChangeWidth = useCallback(
    (width: number) => {
      dispatch(setBoundingBoxDimensions({ width }, optimalDimension));
    },
    [dispatch, optimalDimension]
  );

  const onChangeHeight = useCallback(
    (height: number) => {
      dispatch(setBoundingBoxDimensions({ height }, optimalDimension));
    },
    [dispatch, optimalDimension]
  );

  const onChangeAspectRatioState = useCallback(
    (aspectRatioState: AspectRatioState) => {
      dispatch(aspectRatioChanged(aspectRatioState));
    },
    [dispatch]
  );

  return (
    <ImageSize
      width={width}
      height={height}
      aspectRatioState={aspectRatioState}
      heightComponent={<ParamBoundingBoxHeight />}
      widthComponent={<ParamBoundingBoxWidth />}
      previewComponent={<AspectRatioIconPreview />}
      onChangeAspectRatioState={onChangeAspectRatioState}
      onChangeWidth={onChangeWidth}
      onChangeHeight={onChangeHeight}
    />
  );
});

ImageSizeCanvas.displayName = 'ImageSizeCanvas';
