import { GroupConfig } from 'konva/lib/Group';
import { Group, Line, Rect } from 'react-konva';
import { useAppSelector } from 'app/store';
// import { inpaintingCanvasLinesSelector } from 'features/canvas/canvasSliceSelectors';
import { createSelector } from '@reduxjs/toolkit';
import { currentCanvasSelector, GenericCanvasState } from './canvasSlice';
import { rgbaColorToString } from './util/colorToString';

export const canvasLinesSelector = createSelector(
  currentCanvasSelector,
  (currentCanvas: GenericCanvasState) => {
    const { lines, maskColor, stageCoordinates, stageDimensions, stageScale } =
      currentCanvas;
    return {
      lines,
      stageCoordinates,
      stageDimensions,
      stageScale,
      maskColorString: rgbaColorToString(maskColor),
    };
  }
);

type InpaintingCanvasLinesProps = GroupConfig;

/**
 * Draws the lines which comprise the mask.
 *
 * Uses globalCompositeOperation to handle the brush and eraser tools.
 */
const IAICanvasLines = (props: InpaintingCanvasLinesProps) => {
  const { ...rest } = props;
  const {
    lines,
    maskColorString,
    stageCoordinates,
    stageDimensions,
    stageScale,
  } = useAppSelector(canvasLinesSelector);

  return (
    <Group {...rest}>
      {lines.map((line, i) => (
        <Line
          key={i}
          points={line.points}
          stroke={'rgb(0,0,0)'}
          strokeWidth={line.strokeWidth * 2}
          tension={0}
          lineCap="round"
          lineJoin="round"
          shadowForStrokeEnabled={false}
          listening={false}
          globalCompositeOperation={
            line.tool === 'brush' ? 'source-over' : 'destination-out'
          }
        />
      ))}
      <Rect
        offsetX={stageCoordinates.x / stageScale}
        offsetY={stageCoordinates.y / stageScale}
        height={stageDimensions.height / stageScale}
        width={stageDimensions.width / stageScale}
        fill={maskColorString}
        globalCompositeOperation={'source-in'}
      />
    </Group>
  );
};

export default IAICanvasLines;
