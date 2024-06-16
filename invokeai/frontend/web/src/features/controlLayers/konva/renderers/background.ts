import { getArbitraryBaseColor } from '@invoke-ai/ui-library';
import Konva from 'konva';

const baseGridLineColor = getArbitraryBaseColor(27);
const fineGridLineColor = getArbitraryBaseColor(18);

const getGridSpacing = (scale: number): number => {
  if (scale >= 2) {
    return 8;
  }
  if (scale >= 1 && scale < 2) {
    return 16;
  }
  if (scale >= 0.5 && scale < 1) {
    return 32;
  }
  if (scale >= 0.25 && scale < 0.5) {
    return 64;
  }
  if (scale >= 0.125 && scale < 0.25) {
    return 128;
  }
  return 256;
};

const getBackgroundLayer = (stage: Konva.Stage): Konva.Layer => {
  let background = stage.findOne<Konva.Layer>('#background');
  if (background) {
    return background;
  }

  background = new Konva.Layer({ id: 'background' });
  stage.add(background);
  return background;
};

export const renderBackgroundLayer = (stage: Konva.Stage): void => {
  const background = getBackgroundLayer(stage);
  background.zIndex(0);
  const scale = stage.scaleX();
  const gridSpacing = getGridSpacing(scale);
  const x = stage.x();
  const y = stage.y();
  const width = stage.width();
  const height = stage.height();
  const stageRect = {
    x1: 0,
    y1: 0,
    x2: width,
    y2: height,
    offset: {
      x: x / scale,
      y: y / scale,
    },
  };

  const gridOffset = {
    x: Math.ceil(x / scale / gridSpacing) * gridSpacing,
    y: Math.ceil(y / scale / gridSpacing) * gridSpacing,
  };

  const gridRect = {
    x1: -gridOffset.x,
    y1: -gridOffset.y,
    x2: width / scale - gridOffset.x + gridSpacing,
    y2: height / scale - gridOffset.y + gridSpacing,
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

  const strokeWidth = 1 / scale;
  let _x = 0;
  let _y = 0;

  background.destroyChildren();

  for (let i = 0; i < xSteps; i++) {
    _x = gridFullRect.x1 + i * gridSpacing;
    background.add(
      new Konva.Line({
        x: _x,
        y: gridFullRect.y1,
        points: [0, 0, 0, ySize],
        stroke: _x % 64 ? fineGridLineColor : baseGridLineColor,
        strokeWidth,
        listening: false,
      })
    );
  }
  for (let i = 0; i < ySteps; i++) {
    _y = gridFullRect.y1 + i * gridSpacing;
    background.add(
      new Konva.Line({
        x: gridFullRect.x1,
        y: _y,
        points: [0, 0, xSize, 0],
        stroke: _y % 64 ? fineGridLineColor : baseGridLineColor,
        strokeWidth,
        listening: false,
      })
    );
  }
};
