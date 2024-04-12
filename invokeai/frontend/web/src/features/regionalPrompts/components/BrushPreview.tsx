import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { rgbaColorToString } from 'features/canvas/util/colorToString';
import { $cursorPosition } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo } from 'react';
import { Circle, Group } from 'react-konva';

export const BrushPreviewOutline = memo(() => {
  const brushSize = useAppSelector((s) => s.regionalPrompts.brushSize);
  const tool = useAppSelector((s) => s.regionalPrompts.tool);
  const a = useAppSelector((s) => s.regionalPrompts.promptLayerOpacity);
  const color = useAppSelector((s) => {
    const _color = s.regionalPrompts.layers.find((l) => l.id === s.regionalPrompts.selectedLayer)?.color;
    if (!_color) {
      return null;
    }
    return rgbaColorToString({ ..._color, a });
  });

  const pos = useStore($cursorPosition);

  if (!brushSize || !color || !pos || tool === 'move') {
    return null;
  }

  return (
    <Group listening={false}>
      <Circle
        x={pos.x}
        y={pos.y}
        radius={brushSize / 2}
        fill={color}
        globalCompositeOperation={tool === 'brush' ? 'source-over' : 'destination-out'}
        listening={false}
      />
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
});

BrushPreviewOutline.displayName = 'BrushPreviewOutline';
