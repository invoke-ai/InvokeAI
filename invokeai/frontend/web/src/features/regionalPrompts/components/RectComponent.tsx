import { rgbaColorToString } from 'features/canvas/util/colorToString';
import { useTransform } from 'features/regionalPrompts/hooks/useTransform';
import type { FillRectObject } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { Rect } from 'react-konva';

type Props = {
  rect: FillRectObject;
};

export const RectComponent = ({ rect }: Props) => {
  const { shapeRef } = useTransform(rect);

  return (
    <Rect
      ref={shapeRef}
      key={rect.id}
      x={rect.x}
      y={rect.y}
      width={rect.width}
      height={rect.height}
      fill={rgbaColorToString(rect.color)}
      draggable
    />
  );
};
