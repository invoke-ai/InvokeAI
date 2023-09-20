import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { ImageConfig } from 'konva/lib/shapes/Image';
import { isEqual } from 'lodash-es';
import { memo, useEffect, useState } from 'react';
import { Image as KonvaImage } from 'react-konva';

const selector = createSelector(
  [stateSelector],
  ({ system, canvas }) => {
    const { denoiseProgress } = system;
    const { boundingBox } = canvas.layerState.stagingArea;
    const { sessionIds } = canvas;

    return {
      boundingBox,
      progressImage:
        denoiseProgress && sessionIds.includes(denoiseProgress.session_id)
          ? denoiseProgress.progress_image
          : undefined,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

type Props = Omit<ImageConfig, 'image'>;

const IAICanvasIntermediateImage = (props: Props) => {
  const { ...rest } = props;
  const { progressImage, boundingBox } = useAppSelector(selector);
  const [loadedImageElement, setLoadedImageElement] =
    useState<HTMLImageElement | null>(null);

  useEffect(() => {
    if (!progressImage) {
      return;
    }

    const tempImage = new Image();

    tempImage.onload = () => {
      setLoadedImageElement(tempImage);
    };

    tempImage.src = progressImage.dataURL;
  }, [progressImage]);

  if (!(progressImage && boundingBox)) {
    return null;
  }

  return loadedImageElement ? (
    <KonvaImage
      x={boundingBox.x}
      y={boundingBox.y}
      width={boundingBox.width}
      height={boundingBox.height}
      image={loadedImageElement}
      listening={false}
      {...rest}
    />
  ) : null;
};

export default memo(IAICanvasIntermediateImage);
