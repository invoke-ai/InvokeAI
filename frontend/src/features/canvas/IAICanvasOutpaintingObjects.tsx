import { createSelector } from '@reduxjs/toolkit';
import { RootState, useAppSelector } from 'app/store';
import _ from 'lodash';
import { Group, Line } from 'react-konva';
import { CanvasState } from './canvasSlice';
import IAICanvasImage from './IAICanvasImage';

const selector = createSelector(
  [(state: RootState) => state.canvas],
  (canvas: CanvasState) => {
    return {
      objects:
        canvas.currentCanvas === 'outpainting'
          ? canvas.outpainting.objects
          : undefined,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const IAICanvasOutpaintingObjects = () => {
  const { objects } = useAppSelector(selector);

  if (!objects) return null;

  return (
    <Group name="outpainting-objects">
      {objects.map((obj, i) => {
        if (obj.type === 'image') {
          return (
            <IAICanvasImage key={i} x={obj.x} y={obj.y} url={obj.image.url} />
          );
        } else if (obj.type === 'eraserLine') {
          return (
            <Line
              key={i}
              points={obj.points}
              stroke={'rgb(0,0,0)'} // The lines can be any color, just need alpha > 0
              strokeWidth={obj.strokeWidth * 2}
              tension={0}
              lineCap="round"
              lineJoin="round"
              shadowForStrokeEnabled={false}
              listening={false}
              globalCompositeOperation={'destination-out'}
            />
          );
        }
      })}
    </Group>
  );
};

export default IAICanvasOutpaintingObjects;
