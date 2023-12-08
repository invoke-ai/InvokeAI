import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { isCanvasMaskLine } from 'features/canvas/store/canvasTypes';
import { GroupConfig } from 'konva/lib/Group';
import { memo } from 'react';
import { Group, Line } from 'react-konva';

export const canvasLinesSelector = createMemoizedSelector(
  [stateSelector],
  ({ canvas }) => {
    return { objects: canvas.layerState.objects };
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
  const { objects } = useAppSelector(canvasLinesSelector);

  return (
    <Group listening={false} {...rest}>
      {objects.filter(isCanvasMaskLine).map((line, i) => (
        <Line
          key={i}
          points={line.points}
          stroke="rgb(0,0,0)" // The lines can be any color, just need alpha > 0
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
    </Group>
  );
};

export default memo(IAICanvasLines);
