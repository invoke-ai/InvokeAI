import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import type { CanvasV2State, RgbaColor } from 'features/controlLayers/store/types';
import type { ParameterCanvasCoherenceMode } from 'features/parameters/types/parameterSchemas';

export const compositingReducers = {
  setInfillMethod: (state, action: PayloadAction<string>) => {
    state.compositing.infillMethod = action.payload;
  },
  setInfillTileSize: (state, action: PayloadAction<number>) => {
    state.compositing.infillTileSize = action.payload;
  },
  setInfillPatchmatchDownscaleSize: (state, action: PayloadAction<number>) => {
    state.compositing.infillPatchmatchDownscaleSize = action.payload;
  },
  setInfillColorValue: (state, action: PayloadAction<RgbaColor>) => {
    state.compositing.infillColorValue = action.payload;
  },
  setMaskBlur: (state, action: PayloadAction<number>) => {
    state.compositing.maskBlur = action.payload;
  },
  setCanvasCoherenceMode: (state, action: PayloadAction<ParameterCanvasCoherenceMode>) => {
    state.compositing.canvasCoherenceMode = action.payload;
  },
  setCanvasCoherenceEdgeSize: (state, action: PayloadAction<number>) => {
    state.compositing.canvasCoherenceEdgeSize = action.payload;
  },
  setCanvasCoherenceMinDenoise: (state, action: PayloadAction<number>) => {
    state.compositing.canvasCoherenceMinDenoise = action.payload;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
