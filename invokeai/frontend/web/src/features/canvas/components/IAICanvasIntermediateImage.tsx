import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { systemSelector } from 'features/system/store/systemSelectors';
import { ImageConfig } from 'konva/lib/shapes/Image';
import { isEqual } from 'lodash-es';

import { memo, useEffect, useState } from 'react';
import { Image as KonvaImage } from 'react-konva';
import { canvasSelector } from '../store/canvasSelectors';

const selector = createSelector(
  [systemSelector, canvasSelector],
  (system, canvas) => {
    const { progressImage, sessionId } = system;
    const { sessionId: canvasSessionId, boundingBox } =
      canvas.layerState.stagingArea;

    return {
      boundingBox,
      progressImage: sessionId === canvasSessionId ? progressImage : undefined,
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
