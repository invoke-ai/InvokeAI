import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store';
import { RectConfig } from 'konva/lib/shapes/Rect';
import { Rect } from 'react-konva';
import {
  currentCanvasSelector,
  InpaintingCanvasState,
  OutpaintingCanvasState,
} from './canvasSlice';
import { rgbaColorToString } from './util/colorToString';

export const canvasMaskCompositerSelector = createSelector(
  currentCanvasSelector,
  (currentCanvas) => {
    const { maskColor, stageCoordinates, stageDimensions, stageScale } =
      currentCanvas as InpaintingCanvasState | OutpaintingCanvasState;

    return {
      stageCoordinates,
      stageDimensions,
      stageScale,
      maskColorString: rgbaColorToString(maskColor),
    };
  }
);

type IAICanvasMaskCompositerProps = RectConfig;

const IAICanvasMaskCompositer = (props: IAICanvasMaskCompositerProps) => {
  const { ...rest } = props;

  const { maskColorString, stageCoordinates, stageDimensions, stageScale } =
    useAppSelector(canvasMaskCompositerSelector);

  return (
    <Rect
      offsetX={stageCoordinates.x / stageScale}
      offsetY={stageCoordinates.y / stageScale}
      height={stageDimensions.height / stageScale}
      width={stageDimensions.width / stageScale}
      fill={maskColorString}
      globalCompositeOperation={'source-over'}
      listening={false}
      {...rest}
    />
  );
};

export default IAICanvasMaskCompositer;
