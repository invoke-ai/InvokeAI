import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { aspectRatioChanged, heightChanged, widthChanged } from 'features/controlLayers/store/controlLayersSlice';
import { ParamHeight } from 'features/parameters/components/Core/ParamHeight';
import { ParamWidth } from 'features/parameters/components/Core/ParamWidth';
import { AspectRatioCanvasPreview } from 'features/parameters/components/ImageSize/AspectRatioCanvasPreview';
import { AspectRatioIconPreview } from 'features/parameters/components/ImageSize/AspectRatioIconPreview';
import { ImageSize } from 'features/parameters/components/ImageSize/ImageSize';
import type { AspectRatioState } from 'features/parameters/components/ImageSize/types';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo, useCallback } from 'react';

export const ImageSizeLinear = memo(() => {
  const dispatch = useAppDispatch();
  const tab = useAppSelector(activeTabNameSelector);
  const width = useAppSelector((s) => s.controlLayers.present.size.width);
  const height = useAppSelector((s) => s.controlLayers.present.size.height);
  const aspectRatioState = useAppSelector((s) => s.controlLayers.present.size.aspectRatio);

  const onChangeWidth = useCallback(
    (width: number) => {
      dispatch(widthChanged({ width }));
    },
    [dispatch]
  );

  const onChangeHeight = useCallback(
    (height: number) => {
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
      previewComponent={tab === 'txt2img' ? <AspectRatioCanvasPreview /> : <AspectRatioIconPreview />}
      onChangeAspectRatioState={onChangeAspectRatioState}
      onChangeWidth={onChangeWidth}
      onChangeHeight={onChangeHeight}
    />
  );
});

ImageSizeLinear.displayName = 'ImageSizeLinear';
