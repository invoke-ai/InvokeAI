import { CanvasState } from '../canvasTypes';
import _ from 'lodash';
import { mergeAndUploadCanvas } from '../../util/mergeAndUploadCanvas';
import { uploadImage } from 'features/gallery/util/uploadImage';
import { ActionReducerMapBuilder } from '@reduxjs/toolkit';
import { setInitialCanvasImage } from './setInitialCanvasImage';

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
      setInitialCanvasImage(state, image);
    }
  });
};
