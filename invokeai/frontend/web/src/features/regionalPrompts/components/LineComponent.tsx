import { rgbaColorToString } from 'features/canvas/util/colorToString';
import { useTransform } from 'features/regionalPrompts/hooks/useTransform';
import type { LineObject } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { Line } from 'react-konva';

type Props = {
  line: LineObject;
};

export const LineComponent = ({ line }: Props) => {
  const { shapeRef } = useTransform(line);

  return (
    <Line
      ref={shapeRef}
      key={line.id}
      points={line.points}
      stroke={rgbaColorToString(line.color)}
      strokeWidth={line.strokeWidth}
      draggable
    />
  );
};
