import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  aspectRatioChanged,
  setBoundingBoxDimensions,
} from 'features/canvas/store/canvasSlice';
import ParamBoundingBoxHeight from 'features/parameters/components/Canvas/BoundingBox/ParamBoundingBoxHeight';
import ParamBoundingBoxWidth from 'features/parameters/components/Canvas/BoundingBox/ParamBoundingBoxWidth';
import { ImageSize } from 'features/parameters/components/ImageSize/ImageSize';
import type { AspectRatioState } from 'features/parameters/components/ImageSize/types';
import { memo, useCallback } from 'react';

export const ImageSizeCanvas = memo(() => {
  const dispatch = useAppDispatch();
  const { width, height } = useAppSelector(
    (state) => state.canvas.boundingBoxDimensions
  );
  const aspectRatioState = useAppSelector(
    (state) => state.canvas.aspectRatio
  );

  const onChangeWidth = useCallback(
    (width: number) => {
      dispatch(setBoundingBoxDimensions({ width }));
    },
    [dispatch]
  );

  const onChangeHeight = useCallback(
    (height: number) => {
      dispatch(setBoundingBoxDimensions({ height }));
    },
    [dispatch]
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
      onChangeAspectRatioState={onChangeAspectRatioState}
      onChangeWidth={onChangeWidth}
      onChangeHeight={onChangeHeight}
    />
  );
});

ImageSizeCanvas.displayName = 'ImageSizeCanvas';
