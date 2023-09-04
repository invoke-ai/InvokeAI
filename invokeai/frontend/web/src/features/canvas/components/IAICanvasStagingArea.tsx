import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { GroupConfig } from 'konva/lib/Group';
import { isEqual } from 'lodash-es';

import { Group, Rect } from 'react-konva';
import IAICanvasImage from './IAICanvasImage';
import { memo } from 'react';

const selector = createSelector(
  [canvasSelector],
  (canvas) => {
    const {
      layerState,
      shouldShowStagingImage,
      shouldShowStagingOutline,
      boundingBoxCoordinates: { x, y },
      boundingBoxDimensions: { width, height },
    } = canvas;

    const { selectedImageIndex, images } = layerState.stagingArea;

    return {
      currentStagingAreaImage:
        images.length > 0 && selectedImageIndex !== undefined
          ? images[selectedImageIndex]
          : undefined,
      isOnFirstImage: selectedImageIndex === 0,
      isOnLastImage: selectedImageIndex === images.length - 1,
      shouldShowStagingImage,
      shouldShowStagingOutline,
      x,
      y,
      width,
      height,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

type Props = GroupConfig;

const IAICanvasStagingArea = (props: Props) => {
  const { ...rest } = props;
  const {
    currentStagingAreaImage,
    shouldShowStagingImage,
    shouldShowStagingOutline,
    x,
    y,
    width,
    height,
  } = useAppSelector(selector);

  return (
    <Group {...rest}>
      {shouldShowStagingImage && currentStagingAreaImage && (
        <IAICanvasImage canvasImage={currentStagingAreaImage} />
      )}
      {shouldShowStagingOutline && (
        <Group>
          <Rect
            x={x}
            y={y}
            width={width}
            height={height}
            strokeWidth={1}
            stroke="white"
            strokeScaleEnabled={false}
          />
          <Rect
            x={x}
            y={y}
            width={width}
            height={height}
            dash={[4, 4]}
            strokeWidth={1}
            stroke="black"
            strokeScaleEnabled={false}
          />
        </Group>
      )}
    </Group>
  );
};

export default memo(IAICanvasStagingArea);
