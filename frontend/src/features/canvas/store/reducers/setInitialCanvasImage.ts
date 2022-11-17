import * as InvokeAI from 'app/invokeai';
import { initialLayerState } from '../canvasSlice';
import { CanvasState } from '../canvasTypes';
import {
  roundDownToMultiple,
  roundToMultiple,
} from 'common/util/roundDownToMultiple';
import _ from 'lodash';

export const setInitialCanvasImage = (
  state: CanvasState,
  image: InvokeAI.Image
) => {
  const newBoundingBoxDimensions = {
    width: roundDownToMultiple(_.clamp(image.width, 64, 512), 64),
    height: roundDownToMultiple(_.clamp(image.height, 64, 512), 64),
  };

  const newBoundingBoxCoordinates = {
    x: roundToMultiple(
      image.width / 2 - newBoundingBoxDimensions.width / 2,
      64
    ),
    y: roundToMultiple(
      image.height / 2 - newBoundingBoxDimensions.height / 2,
      64
    ),
  };

  state.boundingBoxDimensions = newBoundingBoxDimensions;
  state.boundingBoxCoordinates = newBoundingBoxCoordinates;

  state.pastLayerStates.push(state.layerState);
  state.layerState = {
    ...initialLayerState,
    objects: [
      {
        kind: 'image',
        layer: 'base',
        x: 0,
        y: 0,
        width: image.width,
        height: image.height,
        image: image,
      },
    ],
  };
  state.futureLayerStates = [];

  state.initialCanvasImageClipRect = {
    clipX: 0,
    clipY: 0,
    clipWidth: image.width,
    clipHeight: image.height,
  };

  state.isCanvasInitialized = false;
  state.doesCanvasNeedScaling = true;
};
