import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { rgbColorToString } from 'features/canvas/util/colorToString';
import { $cursorPosition } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { Circle } from 'react-konva';

export const BrushPreview = () => {
  const brushSize = useAppSelector((s) => s.regionalPrompts.brushSize);
  const color = useAppSelector((s) => {
    const _color = s.regionalPrompts.layers.find((l) => l.id === s.regionalPrompts.selectedLayer)?.color;
    if (!_color) {
      return null;
    }
    return rgbColorToString(_color);
  });
  const pos = useStore($cursorPosition);

  if (!brushSize || !color || !pos) {
    return null;
  }

  return <Circle x={pos.x} y={pos.y} radius={brushSize / 2} fill={color} />;
};
