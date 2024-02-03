import { useShiftModifier } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { roundDownToMultiple, roundDownToMultipleMin, roundToMultiple } from 'common/util/roundDownToMultiple';
import {
  $isDrawing,
  $isMouseOverBoundingBox,
  $isMouseOverBoundingBoxOutline,
  $isMovingBoundingBox,
  $isTransformingBoundingBox,
  $tool,
} from 'features/canvas/store/canvasNanostore';
import {
  aspectRatioChanged,
  setBoundingBoxCoordinates,
  setBoundingBoxDimensions,
  setShouldSnapToGrid,
} from 'features/canvas/store/canvasSlice';
import { CANVAS_GRID_SIZE_COARSE, CANVAS_GRID_SIZE_FINE } from 'features/canvas/store/constants';
import { calculateNewSize } from 'features/parameters/components/ImageSize/calculateNewSize';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import type Konva from 'konva';
import type { GroupConfig } from 'konva/lib/Group';
import type { KonvaEventObject } from 'konva/lib/Node';
import type { Vector2d } from 'konva/lib/types';
import { memo, useCallback, useEffect, useMemo, useRef } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { Group, Rect, Transformer } from 'react-konva';

const borderDash = [4, 4];

type IAICanvasBoundingBoxPreviewProps = GroupConfig;

const IAICanvasBoundingBox = (props: IAICanvasBoundingBoxPreviewProps) => {
  const { ...rest } = props;
  const dispatch = useAppDispatch();
  const boundingBoxCoordinates = useAppSelector((s) => s.canvas.boundingBoxCoordinates);
  const boundingBoxDimensions = useAppSelector((s) => s.canvas.boundingBoxDimensions);
  const stageScale = useAppSelector((s) => s.canvas.stageScale);
  const shouldSnapToGrid = useAppSelector((s) => s.canvas.shouldSnapToGrid);
  const hitStrokeWidth = useAppSelector((s) => 20 / s.canvas.stageScale);
  const aspectRatio = useAppSelector((s) => s.canvas.aspectRatio);
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const transformerRef = useRef<Konva.Transformer>(null);
  const shapeRef = useRef<Konva.Rect>(null);
  const shift = useShiftModifier();
  const tool = useStore($tool);
  const isDrawing = useStore($isDrawing);
  const isMovingBoundingBox = useStore($isMovingBoundingBox);
  const isTransformingBoundingBox = useStore($isTransformingBoundingBox);
  const isMouseOverBoundingBoxOutline = useStore($isMouseOverBoundingBoxOutline);

  useEffect(() => {
    if (!transformerRef.current || !shapeRef.current) {
      return;
    }
    transformerRef.current.nodes([shapeRef.current]);
    transformerRef.current.getLayer()?.batchDraw();
  }, []);

  const gridSize = useMemo(() => (shift ? CANVAS_GRID_SIZE_FINE : CANVAS_GRID_SIZE_COARSE), [shift]);
  const scaledStep = useMemo(() => gridSize * stageScale, [gridSize, stageScale]);

  useHotkeys(
    'N',
    () => {
      dispatch(setShouldSnapToGrid(!shouldSnapToGrid));
    },
    [shouldSnapToGrid]
  );

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

      const newX = roundToMultiple(dragX, gridSize);
      const newY = roundToMultiple(dragY, gridSize);

      e.target.x(newX);
      e.target.y(newY);

      dispatch(
        setBoundingBoxCoordinates({
          x: newX,
          y: newY,
        })
      );
    },
    [dispatch, gridSize, shouldSnapToGrid]
  );

  const handleOnTransform = useCallback(
    (_e: KonvaEventObject<Event>) => {
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

      if (aspectRatio.isLocked) {
        const newDimensions = calculateNewSize(aspectRatio.value, width * height);
        dispatch(
          setBoundingBoxDimensions(
            {
              width: roundDownToMultipleMin(newDimensions.width, gridSize),
              height: roundDownToMultipleMin(newDimensions.height, gridSize),
            },
            optimalDimension
          )
        );
      } else {
        dispatch(
          setBoundingBoxDimensions(
            {
              width: roundDownToMultipleMin(width, gridSize),
              height: roundDownToMultipleMin(height, gridSize),
            },
            optimalDimension
          )
        );
        dispatch(
          aspectRatioChanged({
            isLocked: false,
            id: 'Free',
            value: width / height,
          })
        );
      }

      dispatch(
        setBoundingBoxCoordinates({
          x: shouldSnapToGrid ? roundDownToMultiple(x, gridSize) : x,
          y: shouldSnapToGrid ? roundDownToMultiple(y, gridSize) : y,
        })
      );

      // Reset the scale now that the coords/dimensions have been un-scaled
      rect.scaleX(1);
      rect.scaleY(1);
    },
    [aspectRatio.isLocked, aspectRatio.value, dispatch, shouldSnapToGrid, gridSize, optimalDimension]
  );

  const anchorDragBoundFunc = useCallback(
    (
      oldPos: Vector2d, // old absolute position of anchor point
      newPos: Vector2d, // new absolute position (potentially) of anchor point
      _e: MouseEvent
    ) => {
      /**
       * Konva does not transform with width or height. It transforms the anchor point
       * and scale factor. This is then sent to the shape's onTransform listeners.
       *
       * We need to snap the new dimensions to steps of 8 (or 64). But because the whole
       * stage is scaled, our actual desired step is actually 8 (or 64) * the stage scale.
       *
       * Additionally, we need to ensure we offset the position so that we snap to a
       * multiple of 8 (or 64) that is aligned with the grid, and not from the absolute zero
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

  const handleStartedTransforming = useCallback(() => {
    $isTransformingBoundingBox.set(true);
  }, []);

  const handleEndedTransforming = useCallback(() => {
    $isTransformingBoundingBox.set(false);
    $isMovingBoundingBox.set(false);
    $isMouseOverBoundingBox.set(false);
    $isMouseOverBoundingBoxOutline.set(false);
  }, []);

  const handleStartedMoving = useCallback(() => {
    $isMovingBoundingBox.set(true);
  }, []);

  const handleEndedModifying = useCallback(() => {
    $isTransformingBoundingBox.set(false);
    $isMovingBoundingBox.set(false);
    $isMouseOverBoundingBox.set(false);
    $isMouseOverBoundingBoxOutline.set(false);
  }, []);

  const handleMouseOver = useCallback(() => {
    $isMouseOverBoundingBoxOutline.set(true);
  }, []);

  const handleMouseOut = useCallback(() => {
    !isTransformingBoundingBox && !isMovingBoundingBox && $isMouseOverBoundingBoxOutline.set(false);
  }, [isMovingBoundingBox, isTransformingBoundingBox]);

  const handleMouseEnterBoundingBox = useCallback(() => {
    $isMouseOverBoundingBox.set(true);
  }, []);

  const handleMouseLeaveBoundingBox = useCallback(() => {
    $isMouseOverBoundingBox.set(false);
  }, []);

  const stroke = useMemo(() => {
    if (isMouseOverBoundingBoxOutline || isMovingBoundingBox || isTransformingBoundingBox) {
      return 'rgba(255,255,255,0.5)';
    }
    return 'white';
  }, [isMouseOverBoundingBoxOutline, isMovingBoundingBox, isTransformingBoundingBox]);

  const strokeWidth = useMemo(() => {
    if (isMouseOverBoundingBoxOutline || isMovingBoundingBox || isTransformingBoundingBox) {
      return 6 / stageScale;
    }
    return 1 / stageScale;
  }, [isMouseOverBoundingBoxOutline, isMovingBoundingBox, isTransformingBoundingBox, stageScale]);

  const enabledAnchors = useMemo(() => {
    if (tool !== 'move') {
      return emptyArray;
    }
    if (aspectRatio.isLocked) {
      // TODO: The math to resize the bbox when locked and using other handles is confusing.
      // Workaround for now is to only allow resizing from the bottom-right handle.
      return ['bottom-right'];
    }
    return undefined;
  }, [aspectRatio.isLocked, tool]);

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
        stroke={stroke}
        strokeWidth={strokeWidth}
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
        borderDash={borderDash}
        borderEnabled={true}
        borderStroke="black"
        draggable={false}
        enabledAnchors={enabledAnchors}
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
        shiftBehavior="none"
      />
    </Group>
  );
};

export default memo(IAICanvasBoundingBox);

const emptyArray: string[] = [];
