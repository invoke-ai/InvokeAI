import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { rgbColorToString } from 'features/canvas/util/colorToString';
import { $cursorPosition } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { Circle, Group } from 'react-konva';

export const BrushPreviewFill = () => {
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

export const BrushPreviewOutline = () => {
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

  return (
    <Group>
      <Circle
        x={pos.x}
        y={pos.y}
        radius={brushSize / 2 + 1}
        stroke="rgba(255,255,255,0.8)"
        strokeWidth={1}
        strokeEnabled={true}
        listening={false}
      />
      <Circle
        x={pos.x}
        y={pos.y}
        radius={brushSize / 2}
        stroke="rgba(0,0,0,1)"
        strokeWidth={1}
        strokeEnabled={true}
        listening={false}
      />
    </Group>
  );
};
