import { createSelector } from '@reduxjs/toolkit';
import Konva from 'konva';
import { Context } from 'konva/lib/Context';
import { KonvaEventObject } from 'konva/lib/Node';
import { Box } from 'konva/lib/shapes/Transformer';
import { Vector2d } from 'konva/lib/types';
import _ from 'lodash';
import { useCallback, useEffect, useRef } from 'react';
import { Group, Rect, Transformer } from 'react-konva';
import { useAppDispatch, useAppSelector } from 'app/store';
import { roundToMultiple } from 'common/util/roundDownToMultiple';
import {
  baseCanvasImageSelector,
  currentCanvasSelector,
  GenericCanvasState,
  setBoundingBoxCoordinates,
  setBoundingBoxDimensions,
  setIsMouseOverBoundingBox,
  setIsMovingBoundingBox,
  setIsTransformingBoundingBox,
} from 'features/canvas/canvasSlice';
import { GroupConfig } from 'konva/lib/Group';
import { activeTabNameSelector } from 'features/options/optionsSelectors';

const boundingBoxPreviewSelector = createSelector(
  currentCanvasSelector,
  baseCanvasImageSelector,
  activeTabNameSelector,
  (currentCanvas: GenericCanvasState, baseCanvasImage, activeTabName) => {
    const {
      boundingBoxCoordinates,
      boundingBoxDimensions,
      stageDimensions,
      stageScale,
      shouldLockBoundingBox,
      isDrawing,
      isTransformingBoundingBox,
      isMovingBoundingBox,
      isMouseOverBoundingBox,
      isMoveBoundingBoxKeyHeld,
    } = currentCanvas;
    return {
      boundingBoxCoordinates,
      boundingBoxDimensions,
      isDrawing,
      isMouseOverBoundingBox,
      isMoveBoundingBoxKeyHeld,
      isMovingBoundingBox,
      isTransformingBoundingBox,
      shouldLockBoundingBox,
      stageDimensions,
      stageScale,
      baseCanvasImage,
      activeTabName,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

type IAICanvasBoundingBoxPreviewProps = GroupConfig;

const IAICanvasBoundingBoxPreview = (
  props: IAICanvasBoundingBoxPreviewProps
) => {
  const { ...rest } = props;

  const dispatch = useAppDispatch();
  const {
    boundingBoxCoordinates,
    boundingBoxDimensions,
    isDrawing,
    isMouseOverBoundingBox,
    isMoveBoundingBoxKeyHeld,
    isMovingBoundingBox,
    isTransformingBoundingBox,
    shouldLockBoundingBox,
    stageDimensions,
    stageScale,
    baseCanvasImage,
    activeTabName,
  } = useAppSelector(boundingBoxPreviewSelector);

  const transformerRef = useRef<Konva.Transformer>(null);
  const shapeRef = useRef<Konva.Rect>(null);

  useEffect(() => {
    if (!transformerRef.current || !shapeRef.current) return;
    transformerRef.current.nodes([shapeRef.current]);
    transformerRef.current.getLayer()?.batchDraw();
  }, [shouldLockBoundingBox]);

  const scaledStep = 64 * stageScale;

  const handleOnDragMove = useCallback(
    (e: KonvaEventObject<DragEvent>) => {
      if (activeTabName === 'inpainting') {
        dispatch(
          setBoundingBoxCoordinates({
            x: e.target.x(),
            y: e.target.y(),
          })
        );
        return;
      }

      const dragX = e.target.x();
      const dragY = e.target.y();

      const newX = roundToMultiple(dragX, 64);
      const newY = roundToMultiple(dragY, 64);

      e.target.x(newX);
      e.target.y(newY);

      dispatch(
        setBoundingBoxCoordinates({
          x: newX,
          y: newY,
        })
      );
    },
    [activeTabName, dispatch]
  );

  const dragBoundFunc = useCallback(
    (position: Vector2d) => {
      if (!baseCanvasImage) return boundingBoxCoordinates;

      const { x, y } = position;

      const maxX =
        stageDimensions.width - boundingBoxDimensions.width * stageScale;
      const maxY =
        stageDimensions.height - boundingBoxDimensions.height * stageScale;

      const clampedX = Math.floor(_.clamp(x, 0, maxX));
      const clampedY = Math.floor(_.clamp(y, 0, maxY));

      return { x: clampedX, y: clampedY };
    },
    [
      baseCanvasImage,
      boundingBoxCoordinates,
      stageDimensions.width,
      stageDimensions.height,
      boundingBoxDimensions.width,
      boundingBoxDimensions.height,
      stageScale,
    ]
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
      setBoundingBoxCoordinates({
        x,
        y,
      })
    );

    // Reset the scale now that the coords/dimensions have been un-scaled
    rect.scaleX(1);
    rect.scaleY(1);
  }, [dispatch]);

  // OK
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

  // OK
  const boundBoxFunc = useCallback(
    (oldBoundBox: Box, newBoundBox: Box) => {
      /**
       * The transformer uses this callback to limit valid transformations.
       * Unlike anchorDragBoundFunc, it does get a width and height, so
       * the logic to constrain the size of the bounding box is very simple.
       */
      if (!baseCanvasImage) return oldBoundBox;
      if (
        newBoundBox.width + newBoundBox.x > stageDimensions.width ||
        newBoundBox.height + newBoundBox.y > stageDimensions.height ||
        newBoundBox.x < 0 ||
        newBoundBox.y < 0
      ) {
        return oldBoundBox;
      }

      return newBoundBox;
    },
    [baseCanvasImage, stageDimensions]
  );

  const handleStartedTransforming = (e: KonvaEventObject<MouseEvent>) => {
    e.cancelBubble = true;
    e.evt.stopImmediatePropagation();
    dispatch(setIsTransformingBoundingBox(true));
  };

  const handleEndedTransforming = (e: KonvaEventObject<MouseEvent>) => {
    dispatch(setIsTransformingBoundingBox(false));
    dispatch(setIsMouseOverBoundingBox(false));
  };

  const handleStartedMoving = (e: KonvaEventObject<MouseEvent>) => {
    e.cancelBubble = true;
    e.evt.stopImmediatePropagation();
    dispatch(setIsMovingBoundingBox(true));
  };

  const handleEndedModifying = (e: KonvaEventObject<MouseEvent>) => {
    dispatch(setIsTransformingBoundingBox(false));
    dispatch(setIsMovingBoundingBox(false));
    dispatch(setIsMouseOverBoundingBox(false));
  };

  const spacebarHeldHitFunc = (context: Context, shape: Konva.Shape) => {
    context.rect(0, 0, baseCanvasImage?.width, baseCanvasImage?.height);
    context.fillShape(shape);
  };

  return (
    <Group {...rest}>
      <Rect
        x={boundingBoxCoordinates.x}
        y={boundingBoxCoordinates.y}
        width={boundingBoxDimensions.width}
        height={boundingBoxDimensions.height}
        ref={shapeRef}
        stroke={isMouseOverBoundingBox ? 'rgba(255,255,255,0.3)' : 'white'}
        strokeWidth={Math.floor((isMouseOverBoundingBox ? 8 : 1) / stageScale)}
        fillEnabled={isMoveBoundingBoxKeyHeld}
        hitFunc={isMoveBoundingBoxKeyHeld ? spacebarHeldHitFunc : undefined}
        hitStrokeWidth={Math.floor(13 / stageScale)}
        listening={!isDrawing && !shouldLockBoundingBox}
        onMouseOver={() => {
          dispatch(setIsMouseOverBoundingBox(true));
        }}
        onMouseOut={() => {
          !isTransformingBoundingBox &&
            !isMovingBoundingBox &&
            dispatch(setIsMouseOverBoundingBox(false));
        }}
        onMouseDown={handleStartedMoving}
        onMouseUp={handleEndedModifying}
        draggable={true}
        onDragMove={handleOnDragMove}
        dragBoundFunc={
          activeTabName === 'inpainting' ? dragBoundFunc : undefined
        }
        onTransform={handleOnTransform}
        onDragEnd={handleEndedModifying}
        onTransformEnd={handleEndedTransforming}
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
        draggable={false}
        listening={!isDrawing && !shouldLockBoundingBox}
        onMouseDown={handleStartedTransforming}
        onMouseUp={handleEndedTransforming}
        enabledAnchors={shouldLockBoundingBox ? [] : undefined}
        boundBoxFunc={boundBoxFunc}
        anchorDragBoundFunc={anchorDragBoundFunc}
        onDragEnd={handleEndedModifying}
        onTransformEnd={handleEndedTransforming}
        onMouseOver={() => {
          dispatch(setIsMouseOverBoundingBox(true));
        }}
        onMouseOut={() => {
          !isTransformingBoundingBox &&
            !isMovingBoundingBox &&
            dispatch(setIsMouseOverBoundingBox(false));
        }}
      />
    </Group>
  );
};

export default IAICanvasBoundingBoxPreview;
