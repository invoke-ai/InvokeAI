import { createSelector } from '@reduxjs/toolkit';
import Konva from 'konva';
import _ from 'lodash';
import { useEffect, useRef } from 'react';
import { Group, Rect } from 'react-konva';
import { RootState, useAppSelector } from '../../../../app/store';
import { InpaintingState } from '../inpaintingSlice';
import { rgbaColorToString } from '../util/colorToString';
import { DASH_WIDTH, MARCHING_ANTS_SPEED } from '../util/constants';


const boundingBoxPreviewSelector = createSelector(
  (state: RootState) => state.inpainting,
  (inpainting: InpaintingState) => {
    const {
      boundingBoxCoordinate,
      boundingBoxDimensions,
      boundingBoxPreviewFill,
      canvasDimensions,
      stageScale,
    } = inpainting;
    return {
      boundingBoxCoordinate,
      boundingBoxDimensions,
      boundingBoxPreviewFillString: rgbaColorToString(boundingBoxPreviewFill),
      canvasDimensions,
      dash: DASH_WIDTH / stageScale,  // scale dash lengths
      strokeWidth: 1 / stageScale, // scale stroke thickness
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

/**
 * Shades the area around the mask.
 */
const InpaintingBoundingBoxPreviewOverlay = () => {
  const {
    boundingBoxCoordinate,
    boundingBoxDimensions,
    boundingBoxPreviewFillString,
    canvasDimensions,
  } = useAppSelector(boundingBoxPreviewSelector);

  return (
    <Group>
      <Rect
        x={0}
        y={0}
        height={canvasDimensions.height}
        width={canvasDimensions.width}
        fill={boundingBoxPreviewFillString}
      />
      <Rect
        x={boundingBoxCoordinate.x}
        y={boundingBoxCoordinate.y}
        width={boundingBoxDimensions.width}
        height={boundingBoxDimensions.height}
        fill={'rgb(255,255,255)'}
        listening={false}
        globalCompositeOperation={'destination-out'}
      />
    </Group>
  );
};

/**
 * Draws marching ants around the mask.
 */
const InpaintingBoundingBoxPreviewMarchingAnts = () => {
  const { boundingBoxCoordinate, boundingBoxDimensions } = useAppSelector(
    boundingBoxPreviewSelector
  );

  const blackStrokeRectRef = useRef<Konva.Rect>(null);
  const whiteStrokeRectRef = useRef<Konva.Rect>(null);

  useEffect(() => {
    const blackStrokeRect = blackStrokeRectRef.current;
    const whiteStrokeRect = whiteStrokeRectRef.current;

    const anim = new Konva.Animation((frame) => {
      if (!frame) return;
      blackStrokeRect?.dashOffset(
        -1 * (Math.floor(frame.time / MARCHING_ANTS_SPEED) % 16)
      );
      whiteStrokeRect?.dashOffset(
        -1 * ((Math.floor(frame.time / MARCHING_ANTS_SPEED) % 16) + 4)
      );
    });

    anim.start();

    return () => {
      anim.stop();
    };
  }, []);

  return (
    <Group>
      <Rect
        x={boundingBoxCoordinate.x}
        y={boundingBoxCoordinate.y}
        width={boundingBoxDimensions.width}
        height={boundingBoxDimensions.height}
        stroke={'black'}
        strokeWidth={1}
        dash={[4, 4]}
        ref={blackStrokeRectRef}
        listening={false}
      />
      <Rect
        x={boundingBoxCoordinate.x}
        y={boundingBoxCoordinate.y}
        width={boundingBoxDimensions.width}
        height={boundingBoxDimensions.height}
        stroke={'white'}
        dash={[4, 4]}
        strokeWidth={1}
        ref={whiteStrokeRectRef}
        listening={false}
      />
    </Group>
  );
};

/**
 * Draws non-marching ants around the mask.
 */
const InpaintingBoundingBoxPreviewAnts = () => {
  const { boundingBoxCoordinate, boundingBoxDimensions, dash, strokeWidth } =
    useAppSelector(boundingBoxPreviewSelector);

  return (
    <Group>
      <Rect
        x={boundingBoxCoordinate.x}
        y={boundingBoxCoordinate.y}
        width={boundingBoxDimensions.width}
        height={boundingBoxDimensions.height}
        stroke={'black'}
        strokeWidth={strokeWidth}
        dash={[dash, dash]}
        dashOffset={0}
        listening={false}
      />
      <Rect
        x={boundingBoxCoordinate.x}
        y={boundingBoxCoordinate.y}
        width={boundingBoxDimensions.width}
        height={boundingBoxDimensions.height}
        stroke={'white'}
        dash={[dash, dash]}
        strokeWidth={strokeWidth}
        dashOffset={dash}
        listening={false}
      />
    </Group>
  );
};

const boundingBoxPreviewTypeSelector = createSelector(
  (state: RootState) => state.inpainting,
  (inpainting: InpaintingState) => inpainting.boundingBoxPreviewType
);

const InpaintingBoundingBoxPreview = () => {
  const boundingBoxPreviewType = useAppSelector(boundingBoxPreviewTypeSelector);

  switch (boundingBoxPreviewType) {
    case 'overlay': {
      return <InpaintingBoundingBoxPreviewOverlay />;
    }
    case 'ants': {
      return <InpaintingBoundingBoxPreviewAnts />;
    }
    case 'marchingAnts': {
      return <InpaintingBoundingBoxPreviewMarchingAnts />;
    }
    default:
      return null;
  }
};

export default InpaintingBoundingBoxPreview;
