import { rgbColorToString } from 'features/canvas/util/colorToString';
import type { LineObject } from 'features/regionalPrompts/store/regionalPromptsSlice';
import type { RgbColor } from 'react-colorful';
import { Line } from 'react-konva';

type Props = {
  layerId: string;
  line: LineObject;
  color: RgbColor;
};

export const LineComponent = ({ layerId, line, color }: Props) => {
  return (
    <Line
      id={`layer-${layerId}.line-${line.id}`}
      key={line.id}
      points={line.points}
      strokeWidth={line.strokeWidth}
      stroke={rgbColorToString(color)}
      tension={0}
      lineCap="round"
      lineJoin="round"
      shadowForStrokeEnabled={false}
      globalCompositeOperation={line.tool === 'brush' ? 'source-over' : 'destination-out'}
      listening={false}
    />
  );
};
