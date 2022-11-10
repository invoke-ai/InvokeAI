// Grid drawing adapted from https://longviewcoder.com/2021/12/08/konva-a-better-grid/

import { useColorMode } from '@chakra-ui/react';
import _ from 'lodash';
import { Group, Line as KonvaLine } from 'react-konva';
import useUnscaleCanvasValue from './hooks/useUnscaleCanvasValue';
import { stageRef } from './IAICanvas';

const IAICanvasGrid = () => {
  const { colorMode } = useColorMode();
  const unscale = useUnscaleCanvasValue();

  if (!stageRef.current) return null;
  const gridLineColor =
    colorMode === 'light' ? 'rgba(0,0,0,0.3)' : 'rgba(255,255,255,0.3)';

  const stage = stageRef.current;
  const width = stage.width();
  const height = stage.height();
  const x = stage.x();
  const y = stage.y();

  const stageRect = {
    x1: 0,
    y1: 0,
    x2: width,
    y2: height,
    offset: {
      x: unscale(x),
      y: unscale(y),
    },
  };

  const gridOffset = {
    x: Math.ceil(unscale(x) / 64) * 64,
    y: Math.ceil(unscale(y) / 64) * 64,
  };

  const gridRect = {
    x1: -gridOffset.x,
    y1: -gridOffset.y,
    x2: unscale(width) - gridOffset.x + 64,
    y2: unscale(height) - gridOffset.y + 64,
  };

  const gridFullRect = {
    x1: Math.min(stageRect.x1, gridRect.x1),
    y1: Math.min(stageRect.y1, gridRect.y1),
    x2: Math.max(stageRect.x2, gridRect.x2),
    y2: Math.max(stageRect.y2, gridRect.y2),
  };

  const fullRect = gridFullRect;

  const // find the x & y size of the grid
    xSize = fullRect.x2 - fullRect.x1,
    ySize = fullRect.y2 - fullRect.y1,
    // compute the number of steps required on each axis.
    xSteps = Math.round(xSize / 64) + 1,
    ySteps = Math.round(ySize / 64) + 1;

  return (
    <Group>
      {_.range(0, xSteps).map((i) => (
        <KonvaLine
          key={`x_${i}`}
          x={fullRect.x1 + i * 64}
          y={fullRect.y1}
          points={[0, 0, 0, ySize]}
          stroke={gridLineColor}
          strokeWidth={1}
        />
      ))}
      {_.range(0, ySteps).map((i) => (
        <KonvaLine
          key={`y_${i}`}
          x={fullRect.x1}
          y={fullRect.y1 + i * 64}
          points={[0, 0, xSize, 0]}
          stroke={gridLineColor}
          strokeWidth={1}
        />
      ))}
    </Group>
  );
};

export default IAICanvasGrid;
