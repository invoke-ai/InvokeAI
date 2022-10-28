import { createSelector } from '@reduxjs/toolkit';
import Konva from 'konva';
import { KonvaEventObject } from 'konva/lib/Node';
import { Vector2d } from 'konva/lib/types';
import _ from 'lodash';
import { useEffect, useLayoutEffect, useRef } from 'react';
import { Group, Rect, Transformer } from 'react-konva';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import { roundToMultiple } from '../../../../common/util/roundDownToMultiple';
import {
  InpaintingState,
  setBoundingBoxCoordinate,
  setBoundingBoxDimensions,
  setIsTransformingBoundingBox,
  setIsDrawing,
  setShouldShowBrush,
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
      isMovingBoundingBox,
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
      isMovingBoundingBox,
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

// /**
//  * Draws marching ants around the mask. Unused.
//  */
// const _InpaintingBoundingBoxPreviewMarchingAnts = () => {
//   const { boundingBoxCoordinate, boundingBoxDimensions } = useAppSelector(
//     boundingBoxPreviewSelector
//   );

//   const blackStrokeRectRef = useRef<Konva.Rect>(null);
//   const whiteStrokeRectRef = useRef<Konva.Rect>(null);

//   useEffect(() => {
//     const blackStrokeRect = blackStrokeRectRef.current;
//     const whiteStrokeRect = whiteStrokeRectRef.current;

//     const anim = new Konva.Animation((frame) => {
//       if (!frame) return;
//       blackStrokeRect?.dashOffset(
//         -1 * (Math.floor(frame.time / MARCHING_ANTS_SPEED) % 16)
//       );
//       whiteStrokeRect?.dashOffset(
//         -1 * ((Math.floor(frame.time / MARCHING_ANTS_SPEED) % 16) + 4)
//       );
//     });

//     anim.start();

//     return () => {
//       anim.stop();
//     };
//   }, []);

//   return (
//     <Group>
//       <Rect
//         x={boundingBoxCoordinate.x}
//         y={boundingBoxCoordinate.y}
//         width={boundingBoxDimensions.width}
//         height={boundingBoxDimensions.height}
//         stroke={'black'}
//         strokeWidth={1}
//         dash={[4, 4]}
//         ref={blackStrokeRectRef}
//         listening={false}
//       />
//       <Rect
//         x={boundingBoxCoordinate.x}
//         y={boundingBoxCoordinate.y}
//         width={boundingBoxDimensions.width}
//         height={boundingBoxDimensions.height}
//         stroke={'white'}
//         dash={[4, 4]}
//         strokeWidth={1}
//         ref={whiteStrokeRectRef}
//         listening={false}
//       />
//     </Group>
//   );
// };

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

  const scaledStep = 64 * stageScale;

  console.log(shouldLockBoundingBox);

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
        listening={false}
        onTransformStart={() => {
          dispatch(setIsDrawing(false));
          dispatch(setShouldShowBrush(false));
          dispatch(setIsTransformingBoundingBox(true));
        }}
        onTransformEnd={() => {
          dispatch(setShouldShowBrush(true));
          dispatch(setIsTransformingBoundingBox(false));
        }}
        onTransform={() => {
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
        }}
      />
      <Transformer
        enabledAnchors={shouldLockBoundingBox ? [] : undefined}
        ref={transformerRef}
        rotateEnabled={false}
        anchorSize={15}
        anchorFill={'rgba(212,216,234,1)'}
        anchorStroke={'rgb(42,42,42)'}
        borderEnabled={true}
        borderStroke={'black'}
        borderDash={[4, 4]}
        anchorCornerRadius={3}
        ignoreStroke={true}
        keepRatio={false}
        flipEnabled={false}
        onMouseDown={(e: KonvaEventObject<MouseEvent>) => {
          e.cancelBubble = true;
        }}
        onMouseOver={(e: KonvaEventObject<MouseEvent>) => {
          e.cancelBubble = true;
        }}
        anchorDragBoundFunc={(
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
        }}
        boundBoxFunc={(oldBoundBox, newBoundBox) => {
          /**
           * The transformer uses this callback to limit valid transformations.
           * Unlike anchorDragBoundFunc, it does get a width and height, so
           * the logic to constrain the size of the bounding box is very simple.
           */
          if (!imageToInpaint) return oldBoundBox;

          if (
            newBoundBox.width + newBoundBox.x >
              imageToInpaint.width * stageScale ||
            newBoundBox.height + newBoundBox.y >
              imageToInpaint.height * stageScale ||
            newBoundBox.x < 0 ||
            newBoundBox.y < 0
          ) {
            return oldBoundBox;
          }

          return newBoundBox;
        }}
      />
    </>
  );
};

export default InpaintingBoundingBoxPreview;
