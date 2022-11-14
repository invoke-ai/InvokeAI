import * as InvokeAI from 'app/invokeai';
import { PayloadAction } from '@reduxjs/toolkit';
import { CanvasState, Dimensions, initialLayerState } from './canvasSlice';
import { Vector2d } from 'konva/lib/types';
import { roundDownToMultiple } from 'common/util/roundDownToMultiple';

export const setImageToInpaint_reducer = (
  state: CanvasState,
  image: InvokeAI.Image
  // action: PayloadAction<InvokeAI.Image>
) => {
  const { width: canvasWidth, height: canvasHeight } =
    state.inpainting.stageDimensions;
  const { width, height } = state.inpainting.boundingBoxDimensions;
  const { x, y } = state.inpainting.boundingBoxCoordinates;

  const maxWidth = Math.min(image.width, canvasWidth);
  const maxHeight = Math.min(image.height, canvasHeight);

  const newCoordinates: Vector2d = { x, y };
  const newDimensions: Dimensions = { width, height };

  if (width + x > maxWidth) {
    // Bounding box at least needs to be translated
    if (width > maxWidth) {
      // Bounding box also needs to be resized
      newDimensions.width = roundDownToMultiple(maxWidth, 64);
    }
    newCoordinates.x = maxWidth - newDimensions.width;
  }

  if (height + y > maxHeight) {
    // Bounding box at least needs to be translated
    if (height > maxHeight) {
      // Bounding box also needs to be resized
      newDimensions.height = roundDownToMultiple(maxHeight, 64);
    }
    newCoordinates.y = maxHeight - newDimensions.height;
  }

  state.inpainting.boundingBoxDimensions = newDimensions;
  state.inpainting.boundingBoxCoordinates = newCoordinates;

  state.inpainting.pastLayerStates.push(state.inpainting.layerState);

  state.inpainting.layerState = {
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

  state.outpainting.futureLayerStates = [];
  state.doesCanvasNeedScaling = true;
};

export const setImageToOutpaint_reducer = (
  state: CanvasState,
  image: InvokeAI.Image
) => {
  const { width: canvasWidth, height: canvasHeight } =
    state.outpainting.stageDimensions;
  const { width, height } = state.outpainting.boundingBoxDimensions;
  const { x, y } = state.outpainting.boundingBoxCoordinates;

  const maxWidth = Math.min(image.width, canvasWidth);
  const maxHeight = Math.min(image.height, canvasHeight);

  const newCoordinates: Vector2d = { x, y };
  const newDimensions: Dimensions = { width, height };

  if (width + x > maxWidth) {
    // Bounding box at least needs to be translated
    if (width > maxWidth) {
      // Bounding box also needs to be resized
      newDimensions.width = roundDownToMultiple(maxWidth, 64);
    }
    newCoordinates.x = maxWidth - newDimensions.width;
  }

  if (height + y > maxHeight) {
    // Bounding box at least needs to be translated
    if (height > maxHeight) {
      // Bounding box also needs to be resized
      newDimensions.height = roundDownToMultiple(maxHeight, 64);
    }
    newCoordinates.y = maxHeight - newDimensions.height;
  }

  state.outpainting.boundingBoxDimensions = newDimensions;
  state.outpainting.boundingBoxCoordinates = newCoordinates;

  state.outpainting.pastLayerStates.push(state.outpainting.layerState);
  state.outpainting.layerState = {
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
  state.outpainting.futureLayerStates = [];
  state.doesCanvasNeedScaling = true;
};
