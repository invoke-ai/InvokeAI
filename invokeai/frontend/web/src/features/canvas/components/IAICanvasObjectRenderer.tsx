import { useAppSelector } from 'app/store/storeHooks';
import {
  isCanvasBaseImage,
  isCanvasBaseLine,
  isCanvasEraseRect,
  isCanvasFillRect,
} from 'features/canvas/store/canvasTypes';
import { rgbaColorToString } from 'features/canvas/util/colorToString';
import { memo } from 'react';
import { Group, Line, Rect } from 'react-konva';

import IAICanvasImage from './IAICanvasImage';

const IAICanvasObjectRenderer = () => {
  const objects = useAppSelector((s) => s.canvas.layerState.objects);

  return (
    <Group name="outpainting-objects" listening={false}>
      {objects.map((obj, i) => {
        if (isCanvasBaseImage(obj)) {
          return <IAICanvasImage key={i} canvasImage={obj} />;
        } else if (isCanvasBaseLine(obj)) {
          const line = (
            <Line
              key={i}
              points={obj.points}
              stroke={obj.color ? rgbaColorToString(obj.color) : 'rgb(0,0,0)'} // The lines can be any color, just need alpha > 0
              strokeWidth={obj.strokeWidth * 2}
              tension={0}
              lineCap="round"
              lineJoin="round"
              shadowForStrokeEnabled={false}
              listening={false}
              globalCompositeOperation={obj.tool === 'brush' ? 'source-over' : 'destination-out'}
            />
          );
          if (obj.clip) {
            return (
              <Group
                key={i}
                clipX={obj.clip.x}
                clipY={obj.clip.y}
                clipWidth={obj.clip.width}
                clipHeight={obj.clip.height}
                listening={false}
              >
                {line}
              </Group>
            );
          } else {
            return line;
          }
        } else if (isCanvasFillRect(obj)) {
          return (
            <Rect
              key={i}
              x={obj.x}
              y={obj.y}
              width={obj.width}
              height={obj.height}
              fill={rgbaColorToString(obj.color)}
              listening={false}
            />
          );
        } else if (isCanvasEraseRect(obj)) {
          return (
            <Rect
              key={i}
              x={obj.x}
              y={obj.y}
              width={obj.width}
              height={obj.height}
              fill="rgb(255, 255, 255)"
              globalCompositeOperation="destination-out"
              listening={false}
            />
          );
        }
      })}
    </Group>
  );
};

export default memo(IAICanvasObjectRenderer);
