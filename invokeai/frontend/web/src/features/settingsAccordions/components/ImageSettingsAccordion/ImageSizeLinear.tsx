import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { ParamHeight } from 'features/parameters/components/Core/ParamHeight';
import { ParamWidth } from 'features/parameters/components/Core/ParamWidth';
import { ImageSize } from 'features/parameters/components/ImageSize/ImageSize';
import type { AspectRatioState } from 'features/parameters/components/ImageSize/types';
import { aspectRatioChanged, heightChanged, widthChanged } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';

export const ImageSizeLinear = memo(() => {
  const dispatch = useAppDispatch();
  const width = useAppSelector((s) => s.generation.width);
  const height = useAppSelector((s) => s.generation.height);
  const aspectRatioState = useAppSelector((s) => s.generation.aspectRatio);

  const onChangeWidth = useCallback(
    (width: number) => {
      dispatch(widthChanged(width));
    },
    [dispatch]
  );

  const onChangeHeight = useCallback(
    (height: number) => {
      dispatch(heightChanged(height));
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
      heightComponent={<ParamHeight />}
      widthComponent={<ParamWidth />}
      onChangeAspectRatioState={onChangeAspectRatioState}
      onChangeWidth={onChangeWidth}
      onChangeHeight={onChangeHeight}
    />
  );
});

ImageSizeLinear.displayName = 'ImageSizeLinear';
