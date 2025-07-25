import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { isPlainObject } from 'es-toolkit';
import type { ImageWithDims } from 'features/controlLayers/store/types';
import { zImageWithDims } from 'features/controlLayers/store/types';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { ParameterSpandrelImageToImageModel } from 'features/parameters/types/parameterSchemas';
import type { ControlNetModelConfig } from 'services/api/types';
import { assert } from 'tsafe';
import z from 'zod';

const zUpscaleState = z.object({
  _version: z.literal(2),
  upscaleModel: zModelIdentifierField.nullable(),
  upscaleInitialImage: zImageWithDims.nullable(),
  structure: z.number(),
  creativity: z.number(),
  tileControlnetModel: zModelIdentifierField.nullable(),
  scale: z.number(),
  postProcessingModel: zModelIdentifierField.nullable(),
  tileSize: z.number(),
  tileOverlap: z.number(),
});

export type UpscaleState = z.infer<typeof zUpscaleState>;

const getInitialState = (): UpscaleState => ({
  _version: 2,
  upscaleModel: null,
  upscaleInitialImage: null,
  structure: 0,
  creativity: 0,
  tileControlnetModel: null,
  scale: 4,
  postProcessingModel: null,
  tileSize: 1024,
  tileOverlap: 128,
});

const slice = createSlice({
  name: 'upscale',
  initialState: getInitialState(),
  reducers: {
    upscaleModelChanged: (state, action: PayloadAction<ParameterSpandrelImageToImageModel | null>) => {
      state.upscaleModel = action.payload;
    },
    upscaleInitialImageChanged: (state, action: PayloadAction<ImageWithDims | null>) => {
      state.upscaleInitialImage = action.payload;
    },
    structureChanged: (state, action: PayloadAction<number>) => {
      state.structure = action.payload;
    },
    creativityChanged: (state, action: PayloadAction<number>) => {
      state.creativity = action.payload;
    },
    tileControlnetModelChanged: (state, action: PayloadAction<ControlNetModelConfig | null>) => {
      state.tileControlnetModel = action.payload;
    },
    scaleChanged: (state, action: PayloadAction<number>) => {
      state.scale = action.payload;
    },
    postProcessingModelChanged: (state, action: PayloadAction<ParameterSpandrelImageToImageModel | null>) => {
      state.postProcessingModel = action.payload;
    },
    tileSizeChanged: (state, action: PayloadAction<number>) => {
      state.tileSize = action.payload;
    },
    tileOverlapChanged: (state, action: PayloadAction<number>) => {
      state.tileOverlap = action.payload;
    },
  },
});

export const {
  upscaleModelChanged,
  upscaleInitialImageChanged,
  structureChanged,
  creativityChanged,
  tileControlnetModelChanged,
  scaleChanged,
  postProcessingModelChanged,
  tileSizeChanged,
  tileOverlapChanged,
} = slice.actions;

export const upscaleSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zUpscaleState,
  getInitialState,
  persistConfig: {
    migrate: (state) => {
      assert(isPlainObject(state));
      if (!('_version' in state)) {
        state._version = 1;
      }
      if (state._version === 1) {
        state._version = 2;
        // Migrate from v1 to v2: upscaleInitialImage was an ImageDTO, now it's an ImageWithDims
        if (state.upscaleInitialImage) {
          const { image_name, width, height } = state.upscaleInitialImage;
          state.upscaleInitialImage = {
            image_name,
            width,
            height,
          };
        }
      }
      return zUpscaleState.parse(state);
    },
  },
};

export const selectUpscaleSlice = (state: RootState) => state.upscale;
const createUpscaleSelector = <T>(selector: Selector<UpscaleState, T>) => createSelector(selectUpscaleSlice, selector);
export const selectPostProcessingModel = createUpscaleSelector((upscale) => upscale.postProcessingModel);
export const selectCreativity = createUpscaleSelector((upscale) => upscale.creativity);
export const selectUpscaleModel = createUpscaleSelector((upscale) => upscale.upscaleModel);
export const selectTileControlNetModel = createUpscaleSelector((upscale) => upscale.tileControlnetModel);
export const selectStructure = createUpscaleSelector((upscale) => upscale.structure);
export const selectUpscaleInitialImage = createUpscaleSelector((upscale) => upscale.upscaleInitialImage);
export const selectUpscaleScale = createUpscaleSelector((upscale) => upscale.scale);
export const selectTileSize = createUpscaleSelector((upscale) => upscale.tileSize);
export const selectTileOverlap = createUpscaleSelector((upscale) => upscale.tileOverlap);
