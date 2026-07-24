import type {
  CanvasGradientState,
  CompositeOperation,
  Coordinate,
  Rect,
  RgbaColor,
} from 'features/controlLayers/store/types';

type BuildRadialGradientBufferStateArg = {
  id: string;
  gradientType: 'radial';
  rect: Rect;
  center: Coordinate;
  radius: number;
  clipCenter: Coordinate;
  clipRadius: number;
  clipEnabled: boolean;
  bboxRect: Rect;
  fgColor: RgbaColor;
  bgColor: RgbaColor;
  globalCompositeOperation?: CompositeOperation;
};

type BuildLinearGradientBufferStateArg = {
  id: string;
  gradientType: 'linear';
  rect: Rect;
  start: Coordinate;
  end: Coordinate;
  clipCenter: Coordinate;
  clipRadius: number;
  clipAngle: number;
  clipEnabled: boolean;
  bboxRect: Rect;
  fgColor: RgbaColor;
  bgColor: RgbaColor;
  globalCompositeOperation?: CompositeOperation;
};

type BuildGradientBufferStateArg = BuildRadialGradientBufferStateArg | BuildLinearGradientBufferStateArg;

export const buildGradientBufferState = (arg: BuildGradientBufferStateArg): CanvasGradientState => {
  if (arg.gradientType === 'radial') {
    return {
      id: arg.id,
      type: 'gradient',
      gradientType: 'radial',
      rect: arg.rect,
      center: arg.center,
      radius: Math.max(1, arg.radius),
      clipCenter: arg.clipCenter,
      clipRadius: Math.max(1, arg.clipRadius),
      clipEnabled: arg.clipEnabled,
      bboxRect: arg.bboxRect,
      fgColor: arg.fgColor,
      bgColor: arg.bgColor,
      globalCompositeOperation: arg.globalCompositeOperation,
    };
  }

  const end = arg.end.x === arg.start.x && arg.end.y === arg.start.y ? { x: arg.end.x + 1, y: arg.end.y } : arg.end;

  return {
    id: arg.id,
    type: 'gradient',
    gradientType: 'linear',
    rect: arg.rect,
    start: arg.start,
    end,
    clipCenter: arg.clipCenter,
    clipRadius: Math.max(1, arg.clipRadius),
    clipAngle: arg.clipAngle,
    clipEnabled: arg.clipEnabled,
    bboxRect: arg.bboxRect,
    fgColor: arg.fgColor,
    bgColor: arg.bgColor,
    globalCompositeOperation: arg.globalCompositeOperation,
  };
};
