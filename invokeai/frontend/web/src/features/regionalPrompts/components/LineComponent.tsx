import { rgbColorToString } from 'features/canvas/util/colorToString';
import { useTransform } from 'features/regionalPrompts/hooks/useTransform';
import type { LineObject } from 'features/regionalPrompts/store/regionalPromptsSlice';
import type { RgbColor } from 'react-colorful';
import { Line } from 'react-konva';

type Props = {
  line: LineObject;
  color: RgbColor;
};

export const LineComponent = ({ line, color }: Props) => {
  const { shapeRef } = useTransform(line);

  return (
    <Line
      ref={shapeRef}
      key={line.id}
      points={line.points}
      strokeWidth={line.strokeWidth}
      stroke={rgbColorToString(color)}
      tension={0}
      lineCap="round"
      lineJoin="round"
      shadowForStrokeEnabled={false}
      listening={false}
      globalCompositeOperation={line.tool === 'brush' ? 'source-over' : 'destination-out'}
    />
  );
};
