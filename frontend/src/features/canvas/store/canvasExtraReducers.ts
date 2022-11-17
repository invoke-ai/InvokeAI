import * as InvokeAI from 'app/invokeai';
import { initialLayerState } from './canvasSlice';
import { CanvasState } from './canvasTypes';
import {
  roundDownToMultiple,
  roundToMultiple,
} from 'common/util/roundDownToMultiple';
import _ from 'lodash';
import { mergeAndUploadCanvas } from '../util/mergeAndUploadCanvas';
import { uploadImage } from 'features/gallery/util/uploadImage';
import { ActionReducerMapBuilder } from '@reduxjs/toolkit';

export const setInitialCanvasImage_reducer = (
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

  state.isCanvasInitialized = false;
  state.doesCanvasNeedScaling = true;
};

export const canvasExtraReducers = (
  builder: ActionReducerMapBuilder<CanvasState>
) => {
  builder.addCase(mergeAndUploadCanvas.fulfilled, (state, action) => {
    if (!action.payload) return;
    const { image, kind, originalBoundingBox } = action.payload;

    if (kind === 'temp_merged_canvas') {
      state.pastLayerStates.push({
        ...state.layerState,
      });

      state.futureLayerStates = [];

      state.layerState.objects = [
        {
          kind: 'image',
          layer: 'base',
          ...originalBoundingBox,
          image,
        },
      ];
    }
  });
  builder.addCase(uploadImage.fulfilled, (state, action) => {
    if (!action.payload) return;
    const { image, kind, activeTabName } = action.payload;

    if (kind !== 'init') return;

    if (activeTabName === 'unifiedCanvas') {
      setInitialCanvasImage_reducer(state, image);
    }
  });
};
