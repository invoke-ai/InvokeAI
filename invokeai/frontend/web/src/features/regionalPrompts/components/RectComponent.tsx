import { rgbColorToString } from 'features/canvas/util/colorToString';
import type { FillRectObject } from 'features/regionalPrompts/store/regionalPromptsSlice';
import type { RgbColor } from 'react-colorful';
import { Rect } from 'react-konva';

type Props = {
  rect: FillRectObject;
  color: RgbColor;
};

export const RectComponent = ({ rect, color }: Props) => {
  return (
    <Rect key={rect.id} x={rect.x} y={rect.y} width={rect.width} height={rect.height} fill={rgbColorToString(color)} />
  );
};
