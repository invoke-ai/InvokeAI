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

    const { dataURL, boundingBox: originalBoundingBox } = layerToDataURL(
      canvasImageLayerRef.current,
      stageScale
    );

    if (!dataURL) return;

    const formData = new FormData();

    formData.append(
      'data',
      JSON.stringify({
        dataURL,
        filename: 'merged_canvas.png',
        kind: saveToGallery ? 'result' : 'temp',
        cropVisible: saveToGallery,
      })
    );

    const response = await fetch(window.location.origin + '/upload', {
      method: 'POST',
      body: formData,
    });

    const { url, mtime, width, height } =
      (await response.json()) as InvokeAI.ImageUploadResponse;

    // const newBoundingBox = {
    //   x: bbox[0],
    //   y: bbox[1],
    //   width: bbox[2],
    //   height: bbox[3],
    // };

    const newImage: InvokeAI.Image = {
      uuid: uuidv4(),
      url,
      mtime,
      category: saveToGallery ? 'result' : 'user',
      width: width,
      height: height,
    };

    return {
      image: newImage,
      kind: saveToGallery ? 'merged_canvas' : 'temp_merged_canvas',
      originalBoundingBox,
      // newBoundingBox,
    };
  }
);
