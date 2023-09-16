// Grid drawing adapted from https://longviewcoder.com/2021/12/08/konva-a-better-grid/

import { useColorMode, useToken } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { isEqual, range } from 'lodash-es';

import { ReactNode, memo, useCallback, useLayoutEffect, useState } from 'react';
import { Group, Line as KonvaLine } from 'react-konva';

const selector = createSelector(
  [canvasSelector],
  (canvas) => {
    const { stageScale, stageCoordinates, stageDimensions } = canvas;
    return { stageScale, stageCoordinates, stageDimensions };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const IAICanvasGrid = () => {
  const { stageScale, stageCoordinates, stageDimensions } =
    useAppSelector(selector);
  const { colorMode } = useColorMode();
  const [gridLines, setGridLines] = useState<ReactNode[]>([]);
  const [darkGridLineColor, lightGridLineColor] = useToken('colors', [
    'base.800',
    'base.200',
  ]);

  const unscale = useCallback(
    (value: number) => {
      return value / stageScale;
    },
    [stageScale]
  );

  useLayoutEffect(() => {
    const { width, height } = stageDimensions;
    const { x, y } = stageCoordinates;

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

    const xLines = range(0, xSteps).map((i) => (
      <KonvaLine
        key={`x_${i}`}
        x={fullRect.x1 + i * 64}
        y={fullRect.y1}
        points={[0, 0, 0, ySize]}
        stroke={colorMode === 'dark' ? darkGridLineColor : lightGridLineColor}
        strokeWidth={1}
      />
    ));
    const yLines = range(0, ySteps).map((i) => (
      <KonvaLine
        key={`y_${i}`}
        x={fullRect.x1}
        y={fullRect.y1 + i * 64}
        points={[0, 0, xSize, 0]}
        stroke={colorMode === 'dark' ? darkGridLineColor : lightGridLineColor}
        strokeWidth={1}
      />
    ));

    setGridLines(xLines.concat(yLines));
  }, [
    stageScale,
    stageCoordinates,
    stageDimensions,
    unscale,
    colorMode,
    darkGridLineColor,
    lightGridLineColor,
  ]);

  return <Group>{gridLines}</Group>;
};

export default memo(IAICanvasGrid);
