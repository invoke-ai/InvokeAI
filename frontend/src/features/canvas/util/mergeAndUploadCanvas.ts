import { createAsyncThunk } from '@reduxjs/toolkit';
import { RootState } from 'app/store';
import * as InvokeAI from 'app/invokeai';
import { v4 as uuidv4 } from 'uuid';
import layerToDataURL from './layerToDataURL';
import downloadFile from './downloadFile';
import copyImage from './copyImage';
import { getCanvasBaseLayer } from './konvaInstanceProvider';
import { addToast } from 'features/system/systemSlice';

export const mergeAndUploadCanvas = createAsyncThunk(
  'canvas/mergeAndUploadCanvas',
  async (
    args: {
      cropVisible?: boolean;
      saveToGallery?: boolean;
      downloadAfterSaving?: boolean;
      copyAfterSaving?: boolean;
    },
    thunkAPI
  ) => {
    const { saveToGallery, downloadAfterSaving, cropVisible, copyAfterSaving } =
      args;

    const { getState, dispatch } = thunkAPI;

    const state = getState() as RootState;

    const stageScale = state.canvas.stageScale;

    const canvasBaseLayer = getCanvasBaseLayer();

    if (!canvasBaseLayer) return;

    const { dataURL, boundingBox: originalBoundingBox } = layerToDataURL(
      canvasBaseLayer,
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
        cropVisible,
      })
    );

    const response = await fetch(window.location.origin + '/upload', {
      method: 'POST',
      body: formData,
    });

    const { url, mtime, width, height } =
      (await response.json()) as InvokeAI.ImageUploadResponse;

    if (downloadAfterSaving) {
      downloadFile(url);
      dispatch(
        addToast({
          title: 'Image Download Started',
          status: 'success',
          duration: 2500,
          isClosable: true,
        })
      );
      return;
    }

    if (copyAfterSaving) {
      copyImage(url, width, height);
      dispatch(
        addToast({
          title: 'Image Copied',
          status: 'success',
          duration: 2500,
          isClosable: true,
        })
      );
      return;
    }

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
    };
  }
);
