import { createSelector } from '@reduxjs/toolkit';
import { GroupConfig } from 'konva/lib/Group';
import _ from 'lodash';
import { Circle, Group } from 'react-konva';
import { useAppSelector } from 'app/store';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { rgbaColorToString } from 'features/canvas/util/colorToString';
import {
  COLOR_PICKER_SIZE,
  COLOR_PICKER_STROKE_RADIUS,
} from '../util/constants';

const canvasBrushPreviewSelector = createSelector(
  canvasSelector,
  (canvas) => {
    const {
      cursorPosition,
      stageDimensions: { width, height },
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
    } = canvas;

    return {
      cursorPosition,
      width,
      height,
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
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

/**
 * Draws a black circle around the canvas brush preview.
 */
const IAICanvasToolPreview = (props: GroupConfig) => {
  const { ...rest } = props;
  const {
    cursorPosition,
    width,
    height,
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
  } = useAppSelector(canvasBrushPreviewSelector);

  if (!shouldDrawBrushPreview) return null;

  return (
    <Group listening={false} {...rest}>
      {tool === 'colorPicker' ? (
        <>
          <Circle
            x={cursorPosition ? cursorPosition.x : width / 2}
            y={cursorPosition ? cursorPosition.y : height / 2}
            radius={colorPickerOuterRadius}
            stroke={brushColorString}
            strokeWidth={COLOR_PICKER_STROKE_RADIUS}
            strokeScaleEnabled={false}
          />
          <Circle
            x={cursorPosition ? cursorPosition.x : width / 2}
            y={cursorPosition ? cursorPosition.y : height / 2}
            radius={colorPickerInnerRadius}
            stroke={colorPickerColorString}
            strokeWidth={COLOR_PICKER_STROKE_RADIUS}
            strokeScaleEnabled={false}
          />
        </>
      ) : (
        <>
          <Circle
            x={cursorPosition ? cursorPosition.x : width / 2}
            y={cursorPosition ? cursorPosition.y : height / 2}
            radius={radius}
            fill={layer === 'mask' ? maskColorString : brushColorString}
            globalCompositeOperation={
              tool === 'eraser' ? 'destination-out' : 'source-over'
            }
          />
          <Circle
            x={cursorPosition ? cursorPosition.x : width / 2}
            y={cursorPosition ? cursorPosition.y : height / 2}
            radius={radius}
            stroke={'rgba(255,255,255,0.4)'}
            strokeWidth={strokeWidth * 2}
            strokeEnabled={true}
            listening={false}
          />
          <Circle
            x={cursorPosition ? cursorPosition.x : width / 2}
            y={cursorPosition ? cursorPosition.y : height / 2}
            radius={radius}
            stroke={'rgba(0,0,0,1)'}
            strokeWidth={strokeWidth}
            strokeEnabled={true}
            listening={false}
          />
        </>
      )}
      <Circle
        x={cursorPosition ? cursorPosition.x : width / 2}
        y={cursorPosition ? cursorPosition.y : height / 2}
        radius={dotRadius * 2}
        fill={'rgba(255,255,255,0.4)'}
        listening={false}
      />
      <Circle
        x={cursorPosition ? cursorPosition.x : width / 2}
        y={cursorPosition ? cursorPosition.y : height / 2}
        radius={dotRadius}
        fill={'rgba(0,0,0,1)'}
        listening={false}
      />
    </Group>
  );
};

export default IAICanvasToolPreview;
