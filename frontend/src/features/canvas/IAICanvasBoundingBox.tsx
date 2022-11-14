import { createSelector } from '@reduxjs/toolkit';
import Konva from 'konva';
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
  outpaintingCanvasSelector,
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
  outpaintingCanvasSelector,
  baseCanvasImageSelector,
  activeTabNameSelector,
  (currentCanvas, outpaintingCanvas, baseCanvasImage, activeTabName) => {
    const {
      boundingBoxCoordinates,
      boundingBoxDimensions,
      stageDimensions,
      stageScale,
      isDrawing,
      isTransformingBoundingBox,
      isMovingBoundingBox,
      isMouseOverBoundingBox,
      shouldDarkenOutsideBoundingBox,
      tool,
      stageCoordinates,
    } = currentCanvas;
    const { shouldSnapToGrid } = outpaintingCanvas;

    return {
      boundingBoxCoordinates,
      boundingBoxDimensions,
      isDrawing,
      isMouseOverBoundingBox,
      shouldDarkenOutsideBoundingBox,
      isMovingBoundingBox,
      isTransformingBoundingBox,
      stageDimensions,
      stageScale,
      baseCanvasImage,
      activeTabName,
      shouldSnapToGrid,
      tool,
      stageCoordinates,
      boundingBoxStrokeWidth: (isMouseOverBoundingBox ? 8 : 1) / stageScale,
      hitStrokeWidth: 20 / stageScale,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
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
    isMouseOverBoundingBox,
    shouldDarkenOutsideBoundingBox,
    isMovingBoundingBox,
    isTransformingBoundingBox,
    stageCoordinates,
    stageDimensions,
    stageScale,
    baseCanvasImage,
    activeTabName,
    shouldSnapToGrid,
    tool,
    boundingBoxStrokeWidth,
    hitStrokeWidth,
  } = useAppSelector(boundingBoxPreviewSelector);

  const transformerRef = useRef<Konva.Transformer>(null);
  const shapeRef = useRef<Konva.Rect>(null);

  useEffect(() => {
    if (!transformerRef.current || !shapeRef.current) return;
    transformerRef.current.nodes([shapeRef.current]);
    transformerRef.current.getLayer()?.batchDraw();
  }, []);

  const scaledStep = 64 * stageScale;

  const handleOnDragMove = useCallback(
    (e: KonvaEventObject<DragEvent>) => {
      if (activeTabName === 'inpainting' || !shouldSnapToGrid) {
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
    [activeTabName, dispatch, shouldSnapToGrid]
  );

  const dragBoundFunc = useCallback(
    (position: Vector2d) => {
      if (!baseCanvasImage && activeTabName !== 'outpainting')
        return boundingBoxCoordinates;

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
      activeTabName,
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

  const boundBoxFunc = useCallback(
    (oldBoundBox: Box, newBoundBox: Box) => {
      /**
       * The transformer uses this callback to limit valid transformations.
       * Unlike anchorDragBoundFunc, it does get a width and height, so
       * the logic to constrain the size of the bounding box is very simple.
       */

      // On the Inpainting canvas, the bounding box needs to stay in the stage
      if (
        activeTabName === 'inpainting' &&
        (newBoundBox.width + newBoundBox.x > stageDimensions.width ||
          newBoundBox.height + newBoundBox.y > stageDimensions.height ||
          newBoundBox.x < 0 ||
          newBoundBox.y < 0)
      ) {
        return oldBoundBox;
      }

      return newBoundBox;
    },
    [activeTabName, stageDimensions.height, stageDimensions.width]
  );

  const handleStartedTransforming = () => {
    dispatch(setIsTransformingBoundingBox(true));
  };

  const handleEndedTransforming = () => {
    dispatch(setIsTransformingBoundingBox(false));
    dispatch(setIsMouseOverBoundingBox(false));
  };

  const handleStartedMoving = () => {
    dispatch(setIsMovingBoundingBox(true));
  };

  const handleEndedModifying = () => {
    dispatch(setIsTransformingBoundingBox(false));
    dispatch(setIsMovingBoundingBox(false));
    dispatch(setIsMouseOverBoundingBox(false));
  };

  const handleMouseOver = () => {
    dispatch(setIsMouseOverBoundingBox(true));
  };

  const handleMouseOut = () => {
    !isTransformingBoundingBox &&
      !isMovingBoundingBox &&
      dispatch(setIsMouseOverBoundingBox(false));
  };

  return (
    <Group {...rest}>
      <Rect
        offsetX={stageCoordinates.x / stageScale}
        offsetY={stageCoordinates.y / stageScale}
        height={stageDimensions.height / stageScale}
        width={stageDimensions.width / stageScale}
        fill={'rgba(0,0,0,0.4)'}
        listening={false}
        visible={shouldDarkenOutsideBoundingBox}
      />
      <Rect
        x={boundingBoxCoordinates.x}
        y={boundingBoxCoordinates.y}
        width={boundingBoxDimensions.width}
        height={boundingBoxDimensions.height}
        fill={'rgb(255,255,255)'}
        listening={false}
        visible={shouldDarkenOutsideBoundingBox}
        globalCompositeOperation={'destination-out'}
      />
      <Rect
        {...(activeTabName === 'inpainting' ? { dragBoundFunc } : {})}
        listening={!isDrawing && tool === 'move'}
        draggable={true}
        fillEnabled={false}
        height={boundingBoxDimensions.height}
        onDragEnd={handleEndedModifying}
        onDragMove={handleOnDragMove}
        onMouseDown={handleStartedMoving}
        onMouseOut={handleMouseOut}
        onMouseOver={handleMouseOver}
        onMouseUp={handleEndedModifying}
        onTransform={handleOnTransform}
        onTransformEnd={handleEndedTransforming}
        ref={shapeRef}
        stroke={isMouseOverBoundingBox ? 'rgba(255,255,255,0.7)' : 'white'}
        strokeWidth={boundingBoxStrokeWidth}
        width={boundingBoxDimensions.width}
        x={boundingBoxCoordinates.x}
        y={boundingBoxCoordinates.y}
        hitStrokeWidth={hitStrokeWidth}
      />
      <Transformer
        anchorCornerRadius={3}
        anchorDragBoundFunc={anchorDragBoundFunc}
        anchorFill={'rgba(212,216,234,1)'}
        anchorSize={15}
        anchorStroke={'rgb(42,42,42)'}
        borderDash={[4, 4]}
        borderEnabled={true}
        borderStroke={'black'}
        boundBoxFunc={boundBoxFunc}
        draggable={false}
        enabledAnchors={tool === 'move' ? undefined : []}
        flipEnabled={false}
        ignoreStroke={true}
        keepRatio={false}
        listening={!isDrawing && tool === 'move'}
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

export default IAICanvasBoundingBox;
