import { createSelector } from '@reduxjs/toolkit';
import { GroupConfig } from 'konva/lib/Group';
import _ from 'lodash';
import { Circle, Group, Rect } from 'react-konva';
import { useAppDispatch, useAppSelector } from 'app/store';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { rgbaColorToString } from 'features/canvas/util/colorToString';
import { COLOR_PICKER_SIZE } from '../util/constants';

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

    let fill = '';

    if (layer === 'mask') {
      fill = rgbaColorToString({ ...maskColor, a: 0.5 });
    } else if (tool === 'colorPicker') {
      fill = rgbaColorToString(colorPickerColor);
    } else {
      fill = rgbaColorToString(brushColor);
    }

    return {
      cursorPosition,
      width,
      height,
      radius: brushSize / 2,
      colorPickerSize: COLOR_PICKER_SIZE / stageScale,
      colorPickerOffset: COLOR_PICKER_SIZE / 2 / stageScale,
      colorPickerCornerRadius: COLOR_PICKER_SIZE / 5 / stageScale,
      brushColorString: fill,
      tool,
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
    brushColorString,
    tool,
    shouldDrawBrushPreview,
    dotRadius,
    strokeWidth,
    colorPickerSize,
    colorPickerOffset,
    colorPickerCornerRadius,
  } = useAppSelector(canvasBrushPreviewSelector);

  if (!shouldDrawBrushPreview) return null;

  return (
    <Group listening={false} {...rest}>
      {tool === 'colorPicker' ? (
        <>
          <Rect
            x={
              cursorPosition ? cursorPosition.x - colorPickerOffset : width / 2
            }
            y={
              cursorPosition ? cursorPosition.y - colorPickerOffset : height / 2
            }
            width={colorPickerSize}
            height={colorPickerSize}
            fill={brushColorString}
            cornerRadius={colorPickerCornerRadius}
            listening={false}
          />
          <Rect
            x={
              cursorPosition ? cursorPosition.x - colorPickerOffset : width / 2
            }
            y={
              cursorPosition ? cursorPosition.y - colorPickerOffset : height / 2
            }
            width={colorPickerSize}
            height={colorPickerSize}
            cornerRadius={colorPickerCornerRadius}
            stroke={'rgba(255,255,255,0.4)'}
            strokeWidth={strokeWidth * 2}
            strokeEnabled={true}
            listening={false}
          />
          <Rect
            x={
              cursorPosition ? cursorPosition.x - colorPickerOffset : width / 2
            }
            y={
              cursorPosition ? cursorPosition.y - colorPickerOffset : height / 2
            }
            width={colorPickerSize}
            height={colorPickerSize}
            cornerRadius={colorPickerCornerRadius}
            stroke={'rgba(0,0,0,1)'}
            strokeWidth={strokeWidth}
            strokeEnabled={true}
            listening={false}
          />
        </>
      ) : (
        <>
          <Circle
            x={cursorPosition ? cursorPosition.x : width / 2}
            y={cursorPosition ? cursorPosition.y : height / 2}
            radius={radius}
            fill={brushColorString}
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
