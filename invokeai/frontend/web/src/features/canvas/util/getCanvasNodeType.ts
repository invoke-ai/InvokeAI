import { RootState } from 'app/store/store';
import { getCanvasBaseLayer, getCanvasStage } from './konvaInstanceProvider';
import {
  CanvasObject,
  Dimensions,
  isCanvasMaskLine,
} from '../store/canvasTypes';
import { buildMaskStage, getStageImageData } from './generateMask';
import { log } from 'app/logging/useLogger';
import {
  areAnyPixelsBlack,
  getImageDataTransparency,
} from 'common/util/arrayBuffer';
import { getNodeType } from 'features/nodes/util/getNodeType';
import { Vector2d } from 'konva/lib/types';

const moduleLog = log.child({ namespace: 'getCanvasNodeTypes' });

export type GetCanvasNodeTypeArg = {
  objects: CanvasObject[];
  boundingBoxCoordinates: Vector2d;
  boundingBoxDimensions: Dimensions;
  stageScale: number;
  isMaskEnabled: boolean;
};

export const getCanvasNodeType = (arg: GetCanvasNodeTypeArg) => {
  const canvasBaseLayer = getCanvasBaseLayer();
  const canvasStage = getCanvasStage();

  if (!canvasBaseLayer || !canvasStage) {
    moduleLog.error('Unable to find canvas / stage');
    return;
  }

  const {
    objects,
    boundingBoxCoordinates,
    boundingBoxDimensions,
    stageScale,
    isMaskEnabled,
  } = arg;

  const boundingBox = {
    ...boundingBoxCoordinates,
    ...boundingBoxDimensions,
  };

  const tempScale = canvasBaseLayer.scale();

  canvasBaseLayer.scale({
    x: 1 / stageScale,
    y: 1 / stageScale,
  });

  const absPos = canvasBaseLayer.getAbsolutePosition();

  const scaledBoundingBox = {
    x: boundingBox.x + absPos.x,
    y: boundingBox.y + absPos.y,
    width: boundingBox.width,
    height: boundingBox.height,
  };

  const { stage: maskStage, offscreenContainer } = buildMaskStage(
    isMaskEnabled ? objects.filter(isCanvasMaskLine) : [],
    scaledBoundingBox
  );

  const maskImageData = getStageImageData(maskStage, scaledBoundingBox);

  offscreenContainer.remove();

  if (!maskImageData) {
    moduleLog.error('Unable to get mask stage context');
    return;
  }

  const ctx = canvasBaseLayer.getContext();

  const baseImageData = ctx.getImageData(
    boundingBox.x + absPos.x,
    boundingBox.y + absPos.y,
    boundingBox.width,
    boundingBox.height
  );

  canvasBaseLayer.scale(tempScale);

  const {
    isPartiallyTransparent: baseIsPartiallyTransparent,
    isFullyTransparent: baseIsFullyTransparent,
  } = getImageDataTransparency(baseImageData);

  const doesMaskHaveBlackPixels = areAnyPixelsBlack(maskImageData);

  return getNodeType(
    baseIsPartiallyTransparent,
    baseIsFullyTransparent,
    doesMaskHaveBlackPixels
  );
};
