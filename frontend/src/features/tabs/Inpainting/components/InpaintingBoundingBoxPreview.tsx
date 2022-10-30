import { createSelector } from '@reduxjs/toolkit';
import Konva from 'konva';
import { KonvaEventObject } from 'konva/lib/Node';
import { Box } from 'konva/lib/shapes/Transformer';
import { Vector2d } from 'konva/lib/types';
import _ from 'lodash';
import { useCallback, useEffect, useRef } from 'react';
import { Group, Rect, Transformer } from 'react-konva';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import { roundToMultiple } from '../../../../common/util/roundDownToMultiple';
import { stageRef } from '../InpaintingCanvas';
import {
  InpaintingState,
  setBoundingBoxCoordinate,
  setBoundingBoxDimensions,
} from '../inpaintingSlice';
import { rgbaColorToString } from '../util/colorToString';
import {
  DASH_WIDTH,
  // MARCHING_ANTS_SPEED,
} from '../util/constants';

const boundingBoxPreviewSelector = createSelector(
  (state: RootState) => state.inpainting,
  (inpainting: InpaintingState) => {
    const {
      boundingBoxCoordinate,
      boundingBoxDimensions,
      boundingBoxPreviewFill,
      canvasDimensions,
      stageScale,
      imageToInpaint,
      shouldLockBoundingBox,
    } = inpainting;
    return {
      boundingBoxCoordinate,
      boundingBoxDimensions,
      boundingBoxPreviewFillString: rgbaColorToString(boundingBoxPreviewFill),
      canvasDimensions,
      stageScale,
      imageToInpaint,
      dash: DASH_WIDTH / stageScale, // scale dash lengths
      strokeWidth: 1 / stageScale, // scale stroke thickness
      shouldLockBoundingBox,
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
export const InpaintingBoundingBoxPreviewOverlay = () => {
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

const InpaintingBoundingBoxPreview = () => {
  const dispatch = useAppDispatch();
  const {
    boundingBoxCoordinate,
    boundingBoxDimensions,
    strokeWidth,
    stageScale,
    imageToInpaint,
    shouldLockBoundingBox,
  } = useAppSelector(boundingBoxPreviewSelector);

  const transformerRef = useRef<Konva.Transformer>(null);
  const shapeRef = useRef<Konva.Rect>(null);

  useEffect(() => {
    if (!transformerRef.current || !shapeRef.current) return;
    transformerRef.current.nodes([shapeRef.current]);
    transformerRef.current.getLayer()?.batchDraw();
  }, [shouldLockBoundingBox]);

  useEffect(
    () => () => {
      const container = stageRef.current?.container();
      if (!container) return;
      container.style.cursor = 'none';
    },
    [shouldLockBoundingBox]
  );

  const scaledStep = 64 * stageScale;

  const handleOnDragMove = useCallback(
    (e: KonvaEventObject<DragEvent>) => {
      dispatch(
        setBoundingBoxCoordinate({
          x: Math.floor(e.target.x()),
          y: Math.floor(e.target.y()),
        })
      );
    },
    [dispatch]
  );

  const dragBoundFunc = useCallback(
    (position: Vector2d) => {
      if (!imageToInpaint) return boundingBoxCoordinate;

      const { x, y } = position;

      const maxX = imageToInpaint.width - boundingBoxDimensions.width;
      const maxY = imageToInpaint.height - boundingBoxDimensions.height;

      const clampedX = Math.floor(_.clamp(x, 0, maxX * stageScale));
      const clampedY = Math.floor(_.clamp(y, 0, maxY * stageScale));

      return { x: clampedX, y: clampedY };
    },
    [boundingBoxCoordinate, boundingBoxDimensions, imageToInpaint, stageScale]
  );

  const handleOnTransform = useCallback(() => {
    /**
     * The Konva Transformer changes the object's anchor point and scale factor,
     * not its width and height. We need to un-scale the width and height before
     * setting the values.
     */
    if (!shapeRef.current) return;

    const rect = shapeRef.current;

    const scaleX = rect.scaleX();
    const scaleY = rect.scaleY();

    // undo the scaling
    const width = Math.round(rect.width() * scaleX);
    const height = Math.round(rect.height() * scaleY);

    const x = Math.round(rect.x());
    const y = Math.round(rect.y());

    dispatch(
      setBoundingBoxDimensions({
        width,
        height,
      })
    );

    dispatch(
      setBoundingBoxCoordinate({
        x,
        y,
      })
    );

    // Reset the scale now that the coords/dimensions have been un-scaled
    rect.scaleX(1);
    rect.scaleY(1);
  }, [dispatch]);

  const anchorDragBoundFunc = useCallback(
    (
      oldPos: Vector2d, // old absolute position of anchor point
      newPos: Vector2d, // new absolute position (potentially) of anchor point
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      _e: MouseEvent
    ) => {
      /**
       * Konva does not transform with width or height. It transforms the anchor point
       * and scale factor. This is then sent to the shape's onTransform listeners.
       *
       * We need to snap the new width to steps of 64 without also snapping the
       * coordinates of the bounding box to steps of 64. But because the whole
       * stage is scaled, our actual desired step is actually 64 * the stage scale.
       */

      // Difference of the old coords from the nearest multiple the scaled step
      const offsetX = oldPos.x % scaledStep;
      const offsetY = oldPos.y % scaledStep;

      // Round new position to the nearest multiple of the scaled step
      const closestX = roundToMultiple(newPos.x, scaledStep) + offsetX;
      const closestY = roundToMultiple(newPos.y, scaledStep) + offsetY;

      // the difference between the old coord and new
      const diffX = Math.abs(newPos.x - closestX);
      const diffY = Math.abs(newPos.y - closestY);

      // if the difference is less than the scaled step, we want to snap
      const didSnapX = diffX < scaledStep;
      const didSnapY = diffY < scaledStep;

      // We may not change anything, stash the old position
      let newCoordinate = { ...oldPos };

      // Set the new coords based on what snapped
      if (didSnapX && !didSnapY) {
        newCoordinate = {
          x: closestX,
          y: oldPos.y,
        };
      } else if (!didSnapX && didSnapY) {
        newCoordinate = {
          x: oldPos.x,
          y: closestY,
        };
      } else if (didSnapX && didSnapY) {
        newCoordinate = {
          x: closestX,
          y: closestY,
        };
      }

      return newCoordinate;
    },
    [scaledStep]
  );

  const boundBoxFunc = useCallback(
    (oldBoundBox: Box, newBoundBox: Box) => {
      /**
       * The transformer uses this callback to limit valid transformations.
       * Unlike anchorDragBoundFunc, it does get a width and height, so
       * the logic to constrain the size of the bounding box is very simple.
       */
      if (!imageToInpaint) return oldBoundBox;

      if (
        newBoundBox.width + newBoundBox.x > imageToInpaint.width * stageScale ||
        newBoundBox.height + newBoundBox.y >
          imageToInpaint.height * stageScale ||
        newBoundBox.x < 0 ||
        newBoundBox.y < 0
      ) {
        return oldBoundBox;
      }

      return newBoundBox;
    },
    [imageToInpaint, stageScale]
  );

  return (
    <>
      <Rect
        x={boundingBoxCoordinate.x}
        y={boundingBoxCoordinate.y}
        width={boundingBoxDimensions.width}
        height={boundingBoxDimensions.height}
        ref={shapeRef}
        stroke={'white'}
        strokeWidth={strokeWidth}
        listening={!shouldLockBoundingBox}
        onMouseEnter={(e) => {
          const container = e?.target?.getStage()?.container();
          if (!container) return;
          container.style.cursor = shouldLockBoundingBox ? 'none' : 'move';
        }}
        onMouseLeave={(e) => {
          const container = e?.target?.getStage()?.container();
          if (!container) return;
          container.style.cursor = shouldLockBoundingBox ? 'none' : 'default';
        }}
        draggable={!shouldLockBoundingBox}
        onDragMove={handleOnDragMove}
        dragBoundFunc={dragBoundFunc}
        onTransform={handleOnTransform}
      />
      <Transformer
        ref={transformerRef}
        anchorCornerRadius={3}
        anchorFill={'rgba(212,216,234,1)'}
        anchorSize={15}
        anchorStroke={'rgb(42,42,42)'}
        borderDash={[4, 4]}
        borderStroke={'black'}
        rotateEnabled={false}
        borderEnabled={true}
        flipEnabled={false}
        ignoreStroke={true}
        keepRatio={false}
        listening={!shouldLockBoundingBox}
        enabledAnchors={shouldLockBoundingBox ? [] : undefined}
        boundBoxFunc={boundBoxFunc}
        anchorDragBoundFunc={anchorDragBoundFunc}
      />
    </>
  );
};

export default InpaintingBoundingBoxPreview;
