import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store';
import { GroupConfig } from 'konva/lib/Group';
import _ from 'lodash';
import { Group, Rect } from 'react-konva';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import IAICanvasImage from './IAICanvasImage';

const selector = createSelector(
  [canvasSelector],
  (canvas) => {
    const {
      layerState: {
        stagingArea: { images, selectedImageIndex },
      },
      shouldShowStagingImage,
      shouldShowStagingOutline,
    } = canvas;

    return {
      currentStagingAreaImage:
        images.length > 0 ? images[selectedImageIndex] : undefined,
      isOnFirstImage: selectedImageIndex === 0,
      isOnLastImage: selectedImageIndex === images.length - 1,
      shouldShowStagingImage,
      shouldShowStagingOutline,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
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
  } = useAppSelector(selector);

  if (!currentStagingAreaImage) return null;

  const {
    x,
    y,
    image: { width, height, url },
  } = currentStagingAreaImage;

  return (
    <Group {...rest}>
      {shouldShowStagingImage && <IAICanvasImage url={url} x={x} y={y} />}
      {shouldShowStagingOutline && (
        <Group>
          <Rect
            x={x}
            y={y}
            width={width}
            height={height}
            strokeWidth={1}
            stroke={'black'}
            strokeScaleEnabled={false}
          />
          <Rect
            x={x}
            y={y}
            width={width}
            height={height}
            dash={[4, 4]}
            strokeWidth={1}
            stroke={'white'}
            strokeScaleEnabled={false}
          />
        </Group>
      )}
    </Group>
  );
};

export default IAICanvasStagingArea;
