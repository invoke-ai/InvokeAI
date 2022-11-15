import * as InvokeAI from 'app/invokeai';
import { PayloadAction } from '@reduxjs/toolkit';
import { CanvasState, Dimensions, initialLayerState } from './canvasSlice';
import { Vector2d } from 'konva/lib/types';
import {
  roundDownToMultiple,
  roundToMultiple,
} from 'common/util/roundDownToMultiple';
import _ from 'lodash';

// export const setInitialInpaintingImage = (
//   state: CanvasState,
//   image: InvokeAI.Image
//   // action: PayloadAction<InvokeAI.Image>
// ) => {
//   const { width: canvasWidth, height: canvasHeight } =
//     state.inpainting.stageDimensions;
//   const { width, height } = state.inpainting.boundingBoxDimensions;
//   const { x, y } = state.inpainting.boundingBoxCoordinates;

//   const maxWidth = Math.min(image.width, canvasWidth);
//   const maxHeight = Math.min(image.height, canvasHeight);

//   const newCoordinates: Vector2d = { x, y };
//   const newDimensions: Dimensions = { width, height };

//   if (width + x > maxWidth) {
//     // Bounding box at least needs to be translated
//     if (width > maxWidth) {
//       // Bounding box also needs to be resized
//       newDimensions.width = roundDownToMultiple(maxWidth, 64);
//     }
//     newCoordinates.x = maxWidth - newDimensions.width;
//   }

//   if (height + y > maxHeight) {
//     // Bounding box at least needs to be translated
//     if (height > maxHeight) {
//       // Bounding box also needs to be resized
//       newDimensions.height = roundDownToMultiple(maxHeight, 64);
//     }
//     newCoordinates.y = maxHeight - newDimensions.height;
//   }

//   state.inpainting.boundingBoxDimensions = newDimensions;
//   state.inpainting.boundingBoxCoordinates = newCoordinates;

//   state.inpainting.pastLayerStates.push(state.inpainting.layerState);

//   state.inpainting.layerState = {
//     ...initialLayerState,
//     objects: [
//       {
//         kind: 'image',
//         layer: 'base',
//         x: 0,
//         y: 0,
//         width: image.width,
//         height: image.height,
//         image: image,
//       },
//     ],
//   };

//   state.outpainting.futureLayerStates = [];
//   state.doesCanvasNeedScaling = true;
// };

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

  state.outpainting.boundingBoxDimensions = newBoundingBoxDimensions;

  state.outpainting.boundingBoxCoordinates = newBoundingBoxCoordinates;

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

  state.isCanvasInitialized = false;
  state.doesCanvasNeedScaling = true;
};
