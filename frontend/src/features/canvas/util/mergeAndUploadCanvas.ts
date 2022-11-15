import { createAsyncThunk } from '@reduxjs/toolkit';
import { RootState } from 'app/store';
import Konva from 'konva';
import { MutableRefObject } from 'react';
import * as InvokeAI from 'app/invokeai';
import { v4 as uuidv4 } from 'uuid';
import layerToDataURL from './layerToDataURL';

export const mergeAndUploadCanvas = createAsyncThunk(
  'canvas/mergeAndUploadCanvas',
  async (
    args: {
      canvasImageLayerRef: MutableRefObject<Konva.Layer | null>;
      saveToGallery: boolean;
    },
    thunkAPI
  ) => {
    const { canvasImageLayerRef, saveToGallery } = args;

    const { getState } = thunkAPI;

    const state = getState() as RootState;

    const stageScale = state.canvas[state.canvas.currentCanvas].stageScale;

    if (!canvasImageLayerRef.current) return;

    const { dataURL, relativeX, relativeY } = layerToDataURL(
      canvasImageLayerRef.current,
      stageScale
    );

    if (!dataURL) return;

    const formData = new FormData();

    formData.append('dataURL', dataURL);
    formData.append('filename', 'merged_canvas.png');
    formData.append('kind', saveToGallery ? 'result' : 'temp');

    const response = await fetch(window.location.origin + '/upload', {
      method: 'POST',
      body: formData,
    });

    const { image } = (await response.json()) as InvokeAI.ImageUploadResponse;

    const newImage: InvokeAI.Image = {
      uuid: uuidv4(),
      category: saveToGallery ? 'result' : 'user',
      ...image,
    };

    return {
      image: newImage,
      kind: saveToGallery ? 'merged_canvas' : 'temp_merged_canvas',
      boundingBox: {
        x: relativeX,
        y: relativeY,
        width: image.width,
        height: image.height,
      },
    };
  }
);
