import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import { testImages, testLogs } from './testingData';
import { v4 as uuidv4 } from 'uuid';

// TODO: Split state into more manageable slices, this is getting unwieldy.

export interface SDMetadata {
  prompt: string;
}

export interface SDImage {
  // TODO: I have installed @types/uuid but cannot figure out how to use them here.
  uuid: string;
  url: string;
  metadata: SDMetadata;
}

export interface SDState {
  // settings
  prompt: string;
  imagesToGenerate: number;
  steps: number;
  cfgScale: number;
  height: number;
  width: number;
  sampler: string;
  seed: number;
  img2imgStrength: number;
  gfpganStrength: number;
  upscalingLevel: number;
  upscalingStrength: number;

  // gallery
  currentImageUuid: string;
  images: Array<SDImage>;

  // system
  shouldDisplayInProgress: boolean;
  shouldFitToWidthHeight: boolean;
  isProcessing: boolean;
  progress: number;
  log: Array<string>;
  isGFPGANAvailable: boolean;
  isESRGANAvailable: boolean;
  isConnected: boolean;
}

// Initial state for the main generation parameters
const initialDreamMenuState = {
  prompt: 'Cyborg pickle shooting lasers',
  imagesToGenerate: 1,
  steps: 5,
  cfgScale: 7.5,
  height: 512,
  width: 512,
  sampler: 'k_lms',
  seed: 1,
  img2imgStrength: 0.75,
  gfpganStrength: 0.8,
  upscalingLevel: 0,
  upscalingStrength: 0.75,
};

// Initial state for image viewing
const initialGalleryState = {
  currentImageUuid: '',
  images: testImages,
};

// Initial system state
const initialSystemState = {
  isConnected: false,
  isProcessing: false,
  progress: 0,
  log: testLogs,
  shouldFitToWidthHeight: false,
  shouldDisplayInProgress: false,
  isGFPGANAvailable: true,
  isESRGANAvailable: true,
};

const initialState: SDState = {
  ...initialDreamMenuState,
  ...initialGalleryState,
  ...initialSystemState,
};

export const sdSlice = createSlice({
  name: 'sd',
  initialState,
  reducers: {
    setPrompt: (state, action: PayloadAction<string>) => {
      state.prompt = action.payload;
    },
    setImagesToGenerate: (state, action: PayloadAction<number>) => {
      state.imagesToGenerate = action.payload;
    },
    setSteps: (state, action: PayloadAction<number>) => {
      state.steps = action.payload;
    },
    setCfgScale: (state, action: PayloadAction<number>) => {
      state.cfgScale = action.payload;
    },
    setHeight: (state, action: PayloadAction<number>) => {
      state.height = action.payload;
    },
    setWidth: (state, action: PayloadAction<number>) => {
      state.width = action.payload;
    },
    setSampler: (state, action: PayloadAction<string>) => {
      state.sampler = action.payload;
    },
    setSeed: (state, action: PayloadAction<number>) => {
      state.seed = action.payload;
    },
    setShouldDisplayInProgress: (state, action: PayloadAction<boolean>) => {
      state.shouldDisplayInProgress = action.payload;
    },
    setImg2imgStrength: (state, action: PayloadAction<number>) => {
      state.img2imgStrength = action.payload;
    },
    setShouldFitToWidthHeight: (state, action: PayloadAction<boolean>) => {
      state.shouldFitToWidthHeight = action.payload;
    },
    setGfpganStrength: (state, action: PayloadAction<number>) => {
      state.gfpganStrength = action.payload;
    },
    setUpscalingLevel: (state, action: PayloadAction<number>) => {
      state.upscalingLevel = action.payload;
    },
    setUpscalingStrength: (state, action: PayloadAction<number>) => {
      state.upscalingStrength = action.payload;
    },
    setIsProcessing: (state, action: PayloadAction<boolean>) => {
      state.isProcessing = action.payload;
      state.progress = 0;
    },
    resetSeed: (state) => {
      state.seed = -1;
    },
    resetForm: (state) => {
      return {
        ...state,
        ...initialDreamMenuState,
      };
    },
    setCurrentImage: (state, action: PayloadAction<string>) => {
      const newCurrentImage = state.images.find(
        (image) => image.uuid === action.payload
      );

      if (newCurrentImage) {
        const {
          uuid,
          metadata: { prompt },
        } = newCurrentImage;

        state.currentImageUuid = uuid;
        state.prompt = prompt;
      }
    },
    deleteImage: (state, action: PayloadAction<string>) => {
      const newImages = state.images.filter(
        (image) => image.uuid !== action.payload
      );

      const imageToDeleteIndex = state.images.findIndex(
        (image) => image.uuid === action.payload
      );

      const newCurrentImageIndex = Math.min(
        Math.max(imageToDeleteIndex, 0),
        newImages.length - 1
      );

      state.images = newImages;
      state.currentImageUuid = newImages[newCurrentImageIndex].uuid;
    },
    addImage: (state, action: PayloadAction<SDImage>) => {
      state.images.push(action.payload);
      state.currentImageUuid = action.payload.uuid;
      state.isProcessing = false;
    },
    setProgress: (state, action: PayloadAction<number>) => {
      state.progress = action.payload;
    },
    appendLog: (state, action: PayloadAction<string>) => {
      state.log.push(action.payload);
    },
    setIsConnected: (state, action: PayloadAction<boolean>) => {
      state.isConnected = action.payload;
    },
    setGalleryImages: (state, action: PayloadAction<Array<any>>) => {
      const newGalleryImages = action.payload.map((url) => {
        return {
          uuid: uuidv4(),
          url,
          metadata: {
            prompt: 'test',
          },
        };
      });
      state.images = newGalleryImages;
      state.currentImageUuid =
        newGalleryImages[newGalleryImages.length - 1].uuid;
    },
  },
});

export const {
  setPrompt,
  setImagesToGenerate,
  setSteps,
  setCfgScale,
  setHeight,
  setWidth,
  setSampler,
  setSeed,
  setShouldDisplayInProgress,
  setImg2imgStrength,
  setShouldFitToWidthHeight,
  setGfpganStrength,
  setUpscalingLevel,
  setUpscalingStrength,
  setIsProcessing,
  resetSeed,
  resetForm,
  setCurrentImage,
  deleteImage,
  addImage,
  setProgress,
  appendLog,
  setIsConnected,
  setGalleryImages,
} = sdSlice.actions;

export default sdSlice.reducer;
