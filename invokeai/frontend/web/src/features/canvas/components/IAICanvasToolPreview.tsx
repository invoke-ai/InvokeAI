import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { rgbaColorToString } from 'features/canvas/util/colorToString';
import { GroupConfig } from 'konva/lib/Group';
import { isEqual } from 'lodash-es';

import { Circle, Group } from 'react-konva';
import {
  COLOR_PICKER_SIZE,
  COLOR_PICKER_STROKE_RADIUS,
} from '../util/constants';
import { memo } from 'react';

const canvasBrushPreviewSelector = createSelector(
  canvasSelector,
  (canvas) => {
    const {
      cursorPosition,
      brushSize,
      colorPickerColor,
      maskColor,
      brushColor,
      tool,
      layer,
      shouldShowBrush,
      isMovingBoundingBox,
      isTransformingBoundingBox,
      stageScale,
      stageDimensions,
      boundingBoxCoordinates,
      boundingBoxDimensions,
      shouldRestrictStrokesToBox,
    } = canvas;

    const clip = shouldRestrictStrokesToBox
      ? {
          clipX: boundingBoxCoordinates.x,
          clipY: boundingBoxCoordinates.y,
          clipWidth: boundingBoxDimensions.width,
          clipHeight: boundingBoxDimensions.height,
        }
      : {};

    // // big brain time; this is the *inverse* of the clip that is needed for shouldRestrictStrokesToBox
    // // it took some fiddling to work out, so I am leaving it here in case it is needed for something else...
    // const clipFunc = shouldRestrictStrokesToBox
    //   ? (ctx: SceneContext) => {
    //       console.log(
    //         stageCoordinates.x / stageScale,
    //         stageCoordinates.y / stageScale,
    //         stageDimensions.height / stageScale,
    //         stageDimensions.width / stageScale
    //       );
    //       ctx.fillStyle = 'red';
    //       ctx.rect(
    //         -stageCoordinates.x / stageScale,
    //         -stageCoordinates.y / stageScale,
    //         stageDimensions.width / stageScale,
    //         stageCoordinates.y / stageScale + boundingBoxCoordinates.y
    //       );
    //       ctx.rect(
    //         -stageCoordinates.x / stageScale,
    //         boundingBoxCoordinates.y + boundingBoxDimensions.height,
    //         stageDimensions.width / stageScale,
    //         stageDimensions.height / stageScale
    //       );
    //       ctx.rect(
    //         -stageCoordinates.x / stageScale,
    //         -stageCoordinates.y / stageScale,
    //         stageCoordinates.x / stageScale + boundingBoxCoordinates.x,
    //         stageDimensions.height / stageScale
    //       );
    //       ctx.rect(
    //         boundingBoxCoordinates.x + boundingBoxDimensions.width,
    //         -stageCoordinates.y / stageScale,
    //         stageDimensions.width / stageScale -
    //           (boundingBoxCoordinates.x + boundingBoxDimensions.width),
    //         stageDimensions.height / stageScale
    //       );
    //     }
    //   : undefined;

    return {
      cursorPosition,
      brushX: cursorPosition ? cursorPosition.x : stageDimensions.width / 2,
      brushY: cursorPosition ? cursorPosition.y : stageDimensions.height / 2,
      radius: brushSize / 2,
      colorPickerOuterRadius: COLOR_PICKER_SIZE / stageScale,
      colorPickerInnerRadius:
        (COLOR_PICKER_SIZE - COLOR_PICKER_STROKE_RADIUS + 1) / stageScale,
      maskColorString: rgbaColorToString({ ...maskColor, a: 0.5 }),
      brushColorString: rgbaColorToString(brushColor),
      colorPickerColorString: rgbaColorToString(colorPickerColor),
      tool,
      layer,
      shouldShowBrush,
      shouldDrawBrushPreview:
        !(
          isMovingBoundingBox ||
          isTransformingBoundingBox ||
          !cursorPosition
        ) && shouldShowBrush,
      strokeWidth: 1.5 / stageScale,
      dotRadius: 1.5 / stageScale,
      clip,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

/**
 * Draws a black circle around the canvas brush preview.
 */
const IAICanvasToolPreview = (props: GroupConfig) => {
  const { ...rest } = props;
  const {
    brushX,
    brushY,
    radius,
    maskColorString,
    tool,
    layer,
    shouldDrawBrushPreview,
    dotRadius,
    strokeWidth,
    brushColorString,
    colorPickerColorString,
    colorPickerInnerRadius,
    colorPickerOuterRadius,
    clip,
  } = useAppSelector(canvasBrushPreviewSelector);

  if (!shouldDrawBrushPreview) {
    return null;
  }

  return (
    <Group listening={false} {...clip} {...rest}>
      {tool === 'colorPicker' ? (
        <>
          <Circle
            x={brushX}
            y={brushY}
            radius={colorPickerOuterRadius}
            stroke={brushColorString}
            strokeWidth={COLOR_PICKER_STROKE_RADIUS}
            strokeScaleEnabled={false}
          />
          <Circle
            x={brushX}
            y={brushY}
            radius={colorPickerInnerRadius}
            stroke={colorPickerColorString}
            strokeWidth={COLOR_PICKER_STROKE_RADIUS}
            strokeScaleEnabled={false}
          />
        </>
      ) : (
        <>
          <Circle
            x={brushX}
            y={brushY}
            radius={radius}
            fill={layer === 'mask' ? maskColorString : brushColorString}
            globalCompositeOperation={
              tool === 'eraser' ? 'destination-out' : 'source-out'
            }
          />
          <Circle
            x={brushX}
            y={brushY}
            radius={radius}
            stroke="rgba(255,255,255,0.4)"
            strokeWidth={strokeWidth * 2}
            strokeEnabled={true}
            listening={false}
          />
          <Circle
            x={brushX}
            y={brushY}
            radius={radius}
            stroke="rgba(0,0,0,1)"
            strokeWidth={strokeWidth}
            strokeEnabled={true}
            listening={false}
          />
        </>
      )}
      <Circle
        x={brushX}
        y={brushY}
        radius={dotRadius * 2}
        fill="rgba(255,255,255,0.4)"
        listening={false}
      />
      <Circle
        x={brushX}
        y={brushY}
        radius={dotRadius}
        fill="rgba(0,0,0,1)"
        listening={false}
      />
    </Group>
  );
};

export default memo(IAICanvasToolPreview);
