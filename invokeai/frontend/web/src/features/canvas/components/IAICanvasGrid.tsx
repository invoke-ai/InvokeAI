// Grid drawing adapted from https://longviewcoder.com/2021/12/08/konva-a-better-grid/
import { getArbitraryBaseColor } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasSlice } from 'features/canvas/store/canvasSlice';
import type { ReactElement } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { Group, Line as KonvaLine } from 'react-konva';

const selector = createMemoizedSelector(selectCanvasSlice, (canvas) => {
  return {
    stageCoordinates: canvas.stageCoordinates,
    stageDimensions: canvas.stageDimensions,
  };
});

const baseGridLineColor = getArbitraryBaseColor(27);
const fineGridLineColor = getArbitraryBaseColor(18);

const IAICanvasGrid = () => {
  const { stageCoordinates, stageDimensions } = useAppSelector(selector);
  const stageScale = useAppSelector((s) => s.canvas.stageScale);

  const gridSpacing = useMemo(() => {
    if (stageScale >= 2) {
      return 8;
    }
    if (stageScale >= 1 && stageScale < 2) {
      return 16;
    }
    if (stageScale >= 0.5 && stageScale < 1) {
      return 32;
    }
    return 64;
  }, [stageScale]);

  const unscale = useCallback(
    (value: number) => {
      return value / stageScale;
    },
    [stageScale]
  );

  const gridLines = useMemo(() => {
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
      x: Math.ceil(unscale(x) / gridSpacing) * gridSpacing,
      y: Math.ceil(unscale(y) / gridSpacing) * gridSpacing,
    };

    const gridRect = {
      x1: -gridOffset.x,
      y1: -gridOffset.y,
      x2: unscale(width) - gridOffset.x + gridSpacing,
      y2: unscale(height) - gridOffset.y + gridSpacing,
    };

    const gridFullRect = {
      x1: Math.min(stageRect.x1, gridRect.x1),
      y1: Math.min(stageRect.y1, gridRect.y1),
      x2: Math.max(stageRect.x2, gridRect.x2),
      y2: Math.max(stageRect.y2, gridRect.y2),
    };

    const // find the x & y size of the grid
      xSize = gridFullRect.x2 - gridFullRect.x1;
    const ySize = gridFullRect.y2 - gridFullRect.y1;
    // compute the number of steps required on each axis.
    const xSteps = Math.round(xSize / gridSpacing) + 1;
    const ySteps = Math.round(ySize / gridSpacing) + 1;

    const strokeWidth = unscale(1);

    const gridLines: ReactElement[] = new Array(xSteps + ySteps);
    let _x = 0;
    let _y = 0;
    for (let i = 0; i < xSteps; i++) {
      _x = gridFullRect.x1 + i * gridSpacing;
      gridLines.push(
        <KonvaLine
          key={`x_${i}`}
          x={_x}
          y={gridFullRect.y1}
          points={[0, 0, 0, ySize]}
          stroke={_x % 64 ? fineGridLineColor : baseGridLineColor}
          strokeWidth={strokeWidth}
          listening={false}
        />
      );
    }

    for (let i = 0; i < ySteps; i++) {
      _y = gridFullRect.y1 + i * gridSpacing;
      gridLines.push(
        <KonvaLine
          key={`y_${i}`}
          x={gridFullRect.x1}
          y={_y}
          points={[0, 0, xSize, 0]}
          stroke={_y % 64 ? fineGridLineColor : baseGridLineColor}
          strokeWidth={strokeWidth}
          listening={false}
        />
      );
    }

    return gridLines;
  }, [stageDimensions, stageCoordinates, unscale, gridSpacing]);

  return <Group listening={false}>{gridLines}</Group>;
};

export default memo(IAICanvasGrid);
