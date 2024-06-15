import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { aspectRatioChanged, heightChanged, widthChanged } from 'features/controlLayers/store/canvasV2Slice';
import { ParamHeight } from 'features/parameters/components/Core/ParamHeight';
import { ParamWidth } from 'features/parameters/components/Core/ParamWidth';
import { AspectRatioCanvasPreview } from 'features/parameters/components/ImageSize/AspectRatioCanvasPreview';
import { ImageSize } from 'features/parameters/components/ImageSize/ImageSize';
import type { AspectRatioState } from 'features/parameters/components/ImageSize/types';
import { memo, useCallback } from 'react';

export const ImageSizeLinear = memo(() => {
  const dispatch = useAppDispatch();
  const width = useAppSelector((s) => s.canvasV2.size.width);
  const height = useAppSelector((s) => s.canvasV2.size.height);
  const aspectRatioState = useAppSelector((s) => s.canvasV2.size.aspectRatio);

  const onChangeWidth = useCallback(
    (width: number) => {
      if (width === 0) {
        return;
      }
      dispatch(widthChanged({ width }));
    },
    [dispatch]
  );

  const onChangeHeight = useCallback(
    (height: number) => {
      if (height === 0) {
        return;
      }
      dispatch(heightChanged({ height }));
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
      previewComponent={<AspectRatioCanvasPreview />}
      onChangeAspectRatioState={onChangeAspectRatioState}
      onChangeWidth={onChangeWidth}
      onChangeHeight={onChangeHeight}
    />
  );
});

ImageSizeLinear.displayName = 'ImageSizeLinear';
