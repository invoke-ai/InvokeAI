import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  roundDownToMultiple,
  roundToMultiple,
} from 'common/util/roundDownToMultiple';
import {
  setBoundingBoxCoordinates,
  setBoundingBoxDimensions,
  setIsMouseOverBoundingBox,
  setIsMovingBoundingBox,
  setIsTransformingBoundingBox,
  setShouldSnapToGrid,
} from 'features/canvas/store/canvasSlice';
import Konva from 'konva';
import { GroupConfig } from 'konva/lib/Group';
import { KonvaEventObject } from 'konva/lib/Node';
import { Vector2d } from 'konva/lib/types';
import { isEqual } from 'lodash-es';

import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { Group, Rect, Transformer } from 'react-konva';

const boundingBoxPreviewSelector = createSelector(
  [stateSelector],
  ({ canvas, generation }) => {
    const {
      boundingBoxCoordinates,
      boundingBoxDimensions,
      stageScale,
      isDrawing,
      isTransformingBoundingBox,
      isMovingBoundingBox,
      tool,
      shouldSnapToGrid,
    } = canvas;

    const { aspectRatio } = generation;

    return {
      boundingBoxCoordinates,
      boundingBoxDimensions,
      isDrawing,
      isMovingBoundingBox,
      isTransformingBoundingBox,
      stageScale,
      shouldSnapToGrid,
      tool,
      hitStrokeWidth: 20 / stageScale,
      aspectRatio,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

type IAICanvasBoundingBoxPreviewProps = GroupConfig;

const IAICanvasBoundingBox = (props: IAICanvasBoundingBoxPreviewProps) => {
  const { ...rest } = props;

  const dispatch = useAppDispatch();
  const {
    boundingBoxCoordinates,
    boundingBoxDimensions,
    isDrawing,
    isMovingBoundingBox,
    isTransformingBoundingBox,
    stageScale,
    shouldSnapToGrid,
    tool,
    hitStrokeWidth,
    aspectRatio,
  } = useAppSelector(boundingBoxPreviewSelector);

  const transformerRef = useRef<Konva.Transformer>(null);
  const shapeRef = useRef<Konva.Rect>(null);

  const [isMouseOverBoundingBoxOutline, setIsMouseOverBoundingBoxOutline] =
    useState(false);

  useEffect(() => {
    if (!transformerRef.current || !shapeRef.current) {
      return;
    }
    transformerRef.current.nodes([shapeRef.current]);
    transformerRef.current.getLayer()?.batchDraw();
  }, []);

  const scaledStep = 64 * stageScale;

  useHotkeys('N', () => {
    dispatch(setShouldSnapToGrid(!shouldSnapToGrid));
  });

  const handleOnDragMove = useCallback(
    (e: KonvaEventObject<DragEvent>) => {
      if (!shouldSnapToGrid) {
        dispatch(
          setBoundingBoxCoordinates({
            x: Math.floor(e.target.x()),
            y: Math.floor(e.target.y()),
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
    [dispatch, shouldSnapToGrid]
  );

  const handleOnTransform = useCallback(() => {
    /**
     * The Konva Transformer changes the object's anchor point and scale factor,
     * not its width and height. We need to un-scale the width and height before
     * setting the values.
     */
    if (!shapeRef.current) {
      return;
    }

    const rect = shapeRef.current;

    const scaleX = rect.scaleX();
    const scaleY = rect.scaleY();

    // undo the scaling
    const width = Math.round(rect.width() * scaleX);
    const height = Math.round(rect.height() * scaleY);

    const x = Math.round(rect.x());
    const y = Math.round(rect.y());

    if (aspectRatio) {
      const newHeight = roundToMultiple(width / aspectRatio, 64);
      dispatch(
        setBoundingBoxDimensions({
          width: width,
          height: newHeight,
        })
      );
    } else {
      dispatch(
        setBoundingBoxDimensions({
          width,
          height,
        })
      );
    }

    dispatch(
      setBoundingBoxCoordinates({
        x: shouldSnapToGrid ? roundDownToMultiple(x, 64) : x,
        y: shouldSnapToGrid ? roundDownToMultiple(y, 64) : y,
      })
    );

    // Reset the scale now that the coords/dimensions have been un-scaled
    rect.scaleX(1);
    rect.scaleY(1);
  }, [dispatch, shouldSnapToGrid, aspectRatio]);

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
       * We need to snap the new dimensions to steps of 64. But because the whole
       * stage is scaled, our actual desired step is actually 64 * the stage scale.
       *
       * Additionally, we need to ensure we offset the position so that we snap to a
       * multiple of 64 that is aligned with the grid, and not from the absolute zero
       * coordinate.
       */

      // Calculate the offset of the grid.
      const offsetX = oldPos.x % scaledStep;
      const offsetY = oldPos.y % scaledStep;

      const newCoordinates = {
        x: roundDownToMultiple(newPos.x, scaledStep) + offsetX,
        y: roundDownToMultiple(newPos.y, scaledStep) + offsetY,
      };

      return newCoordinates;
    },
    [scaledStep]
  );

  const handleStartedTransforming = () => {
    dispatch(setIsTransformingBoundingBox(true));
  };

  const handleEndedTransforming = () => {
    dispatch(setIsTransformingBoundingBox(false));
    dispatch(setIsMovingBoundingBox(false));
    dispatch(setIsMouseOverBoundingBox(false));
    setIsMouseOverBoundingBoxOutline(false);
  };

  const handleStartedMoving = () => {
    dispatch(setIsMovingBoundingBox(true));
  };

  const handleEndedModifying = () => {
    dispatch(setIsTransformingBoundingBox(false));
    dispatch(setIsMovingBoundingBox(false));
    dispatch(setIsMouseOverBoundingBox(false));
    setIsMouseOverBoundingBoxOutline(false);
  };

  const handleMouseOver = () => {
    setIsMouseOverBoundingBoxOutline(true);
  };

  const handleMouseOut = () => {
    !isTransformingBoundingBox &&
      !isMovingBoundingBox &&
      setIsMouseOverBoundingBoxOutline(false);
  };

  const handleMouseEnterBoundingBox = () => {
    dispatch(setIsMouseOverBoundingBox(true));
  };

  const handleMouseLeaveBoundingBox = () => {
    dispatch(setIsMouseOverBoundingBox(false));
  };

  return (
    <Group {...rest}>
      <Rect
        height={boundingBoxDimensions.height}
        width={boundingBoxDimensions.width}
        x={boundingBoxCoordinates.x}
        y={boundingBoxCoordinates.y}
        onMouseEnter={handleMouseEnterBoundingBox}
        onMouseOver={handleMouseEnterBoundingBox}
        onMouseLeave={handleMouseLeaveBoundingBox}
        onMouseOut={handleMouseLeaveBoundingBox}
      />
      <Rect
        draggable={true}
        fillEnabled={false}
        height={boundingBoxDimensions.height}
        hitStrokeWidth={hitStrokeWidth}
        listening={!isDrawing && tool === 'move'}
        onDragStart={handleStartedMoving}
        onDragEnd={handleEndedModifying}
        onDragMove={handleOnDragMove}
        onMouseDown={handleStartedMoving}
        onMouseOut={handleMouseOut}
        onMouseOver={handleMouseOver}
        onMouseEnter={handleMouseOver}
        onMouseUp={handleEndedModifying}
        onTransform={handleOnTransform}
        onTransformEnd={handleEndedTransforming}
        ref={shapeRef}
        stroke={
          isMouseOverBoundingBoxOutline ? 'rgba(255,255,255,0.7)' : 'white'
        }
        strokeWidth={(isMouseOverBoundingBoxOutline ? 8 : 1) / stageScale}
        width={boundingBoxDimensions.width}
        x={boundingBoxCoordinates.x}
        y={boundingBoxCoordinates.y}
      />
      <Transformer
        anchorCornerRadius={3}
        anchorDragBoundFunc={anchorDragBoundFunc}
        anchorFill="rgba(212,216,234,1)"
        anchorSize={15}
        anchorStroke="rgb(42,42,42)"
        borderDash={[4, 4]}
        borderEnabled={true}
        borderStroke="black"
        draggable={false}
        enabledAnchors={tool === 'move' ? undefined : []}
        flipEnabled={false}
        ignoreStroke={true}
        keepRatio={false}
        listening={!isDrawing && tool === 'move'}
        onDragStart={handleStartedMoving}
        onDragEnd={handleEndedModifying}
        onMouseDown={handleStartedTransforming}
        onMouseUp={handleEndedTransforming}
        onTransformEnd={handleEndedTransforming}
        ref={transformerRef}
        rotateEnabled={false}
      />
    </Group>
  );
};

export default memo(IAICanvasBoundingBox);
