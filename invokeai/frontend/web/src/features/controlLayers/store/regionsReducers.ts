import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type {
  CanvasV2State,
  CLIPVisionModelV2,
  FillStyle,
  IPMethodV2,
  RegionalGuidanceIPAdapterConfig,
  RgbColor,
} from 'features/controlLayers/store/types';
import { imageDTOToImageWithDims } from 'features/controlLayers/store/types';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { ParameterAutoNegative } from 'features/parameters/types/parameterSchemas';
import { isEqual } from 'lodash-es';
import type { ImageDTO, IPAdapterModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

import type { CanvasRegionalGuidanceState } from './types';

export const selectRegionalGuidanceEntity = (state: CanvasV2State, id: string) => {
  return state.regions.entities.find((rg) => rg.id === id);
};
export const selectRegionalGuidanceIPAdapter = (state: CanvasV2State, id: string, ipAdapterId: string) => {
  const entity = state.regions.entities.find((rg) => rg.id === id);
  if (!entity) {
    return;
  }
  return entity.ipAdapters.find((ipa) => ipa.id === ipAdapterId);
};
export const selectRegionalGuidanceEntityOrThrow = (state: CanvasV2State, id: string) => {
  const rg = selectRegionalGuidanceEntity(state, id);
  assert(rg, `Region with id ${id} not found`);
  return rg;
};

const DEFAULT_MASK_COLORS: RgbColor[] = [
  { r: 121, g: 157, b: 219 }, // rgb(121, 157, 219)
  { r: 131, g: 214, b: 131 }, // rgb(131, 214, 131)
  { r: 250, g: 225, b: 80 }, // rgb(250, 225, 80)
  { r: 220, g: 144, b: 101 }, // rgb(220, 144, 101)
  { r: 224, g: 117, b: 117 }, // rgb(224, 117, 117)
  { r: 213, g: 139, b: 202 }, // rgb(213, 139, 202)
  { r: 161, g: 120, b: 214 }, // rgb(161, 120, 214)
];

const getRGMaskFill = (state: CanvasV2State): RgbColor => {
  const lastFill = state.regions.entities.slice(-1)[0]?.fill.color;
  let i = DEFAULT_MASK_COLORS.findIndex((c) => isEqual(c, lastFill));
  if (i === -1) {
    i = 0;
  }
  i = (i + 1) % DEFAULT_MASK_COLORS.length;
  const fill = DEFAULT_MASK_COLORS[i];
  assert(fill, 'This should never happen');
  return fill;
};

export const regionsReducers = {
  rgAdded: {
    reducer: (state, action: PayloadAction<{ id: string }>) => {
      const { id } = action.payload;
      const rg: CanvasRegionalGuidanceState = {
        id,
        name: null,
        type: 'regional_guidance',
        isEnabled: true,
        objects: [],
        fill: {
          style: 'solid',
          color: getRGMaskFill(state),
        },
        opacity: 0.5,
        position: { x: 0, y: 0 },
        autoNegative: 'invert',
        positivePrompt: '',
        negativePrompt: null,
        ipAdapters: [],
        rasterizationCache: [],
      };
      state.regions.entities.push(rg);
      state.selectedEntityIdentifier = { type: 'regional_guidance', id };
    },
    prepare: () => ({ payload: { id: getPrefixedId('regional_guidance') } }),
  },
  rgRecalled: (state, action: PayloadAction<{ data: CanvasRegionalGuidanceState }>) => {
    const { data } = action.payload;
    state.regions.entities.push(data);
    state.selectedEntityIdentifier = { type: 'regional_guidance', id: data.id };
  },
  rgAllDeleted: (state) => {
    state.regions.entities = [];
  },
  rgPositivePromptChanged: (state, action: PayloadAction<{ id: string; prompt: string | null }>) => {
    const { id, prompt } = action.payload;
    const entity = selectRegionalGuidanceEntity(state, id);
    if (!entity) {
      return;
    }
    entity.positivePrompt = prompt;
  },
  rgNegativePromptChanged: (state, action: PayloadAction<{ id: string; prompt: string | null }>) => {
    const { id, prompt } = action.payload;
    const entity = selectRegionalGuidanceEntity(state, id);
    if (!entity) {
      return;
    }
    entity.negativePrompt = prompt;
  },
  rgFillColorChanged: (state, action: PayloadAction<{ id: string; color: RgbColor }>) => {
    const { id, color } = action.payload;
    const entity = selectRegionalGuidanceEntity(state, id);
    if (!entity) {
      return;
    }
    entity.fill.color = color;
  },
  rgFillStyleChanged: (state, action: PayloadAction<{ id: string; style: FillStyle }>) => {
    const { id, style } = action.payload;
    const entity = selectRegionalGuidanceEntity(state, id);
    if (!entity) {
      return;
    }
    entity.fill.style = style;
  },

  rgAutoNegativeChanged: (state, action: PayloadAction<{ id: string; autoNegative: ParameterAutoNegative }>) => {
    const { id, autoNegative } = action.payload;
    const rg = selectRegionalGuidanceEntity(state, id);
    if (!rg) {
      return;
    }
    rg.autoNegative = autoNegative;
  },
  rgIPAdapterAdded: (state, action: PayloadAction<{ id: string; ipAdapter: RegionalGuidanceIPAdapterConfig }>) => {
    const { id, ipAdapter } = action.payload;
    const entity = selectRegionalGuidanceEntity(state, id);
    if (!entity) {
      return;
    }
    entity.ipAdapters.push(ipAdapter);
  },
  rgIPAdapterDeleted: (state, action: PayloadAction<{ id: string; ipAdapterId: string }>) => {
    const { id, ipAdapterId } = action.payload;
    const entity = selectRegionalGuidanceEntity(state, id);
    if (!entity) {
      return;
    }
    entity.ipAdapters = entity.ipAdapters.filter((ipAdapter) => ipAdapter.id !== ipAdapterId);
  },
  rgIPAdapterImageChanged: (
    state,
    action: PayloadAction<{ id: string; ipAdapterId: string; imageDTO: ImageDTO | null }>
  ) => {
    const { id, ipAdapterId, imageDTO } = action.payload;
    const ipAdapter = selectRegionalGuidanceIPAdapter(state, id, ipAdapterId);
    if (!ipAdapter) {
      return;
    }
    ipAdapter.image = imageDTO ? imageDTOToImageWithDims(imageDTO) : null;
  },
  rgIPAdapterWeightChanged: (state, action: PayloadAction<{ id: string; ipAdapterId: string; weight: number }>) => {
    const { id, ipAdapterId, weight } = action.payload;
    const ipAdapter = selectRegionalGuidanceIPAdapter(state, id, ipAdapterId);
    if (!ipAdapter) {
      return;
    }
    ipAdapter.weight = weight;
  },
  rgIPAdapterBeginEndStepPctChanged: (
    state,
    action: PayloadAction<{ id: string; ipAdapterId: string; beginEndStepPct: [number, number] }>
  ) => {
    const { id, ipAdapterId, beginEndStepPct } = action.payload;
    const ipAdapter = selectRegionalGuidanceIPAdapter(state, id, ipAdapterId);
    if (!ipAdapter) {
      return;
    }
    ipAdapter.beginEndStepPct = beginEndStepPct;
  },
  rgIPAdapterMethodChanged: (state, action: PayloadAction<{ id: string; ipAdapterId: string; method: IPMethodV2 }>) => {
    const { id, ipAdapterId, method } = action.payload;
    const ipAdapter = selectRegionalGuidanceIPAdapter(state, id, ipAdapterId);
    if (!ipAdapter) {
      return;
    }
    ipAdapter.method = method;
  },
  rgIPAdapterModelChanged: (
    state,
    action: PayloadAction<{
      id: string;
      ipAdapterId: string;
      modelConfig: IPAdapterModelConfig | null;
    }>
  ) => {
    const { id, ipAdapterId, modelConfig } = action.payload;
    const ipAdapter = selectRegionalGuidanceIPAdapter(state, id, ipAdapterId);
    if (!ipAdapter) {
      return;
    }
    ipAdapter.model = modelConfig ? zModelIdentifierField.parse(modelConfig) : null;
  },
  rgIPAdapterCLIPVisionModelChanged: (
    state,
    action: PayloadAction<{ id: string; ipAdapterId: string; clipVisionModel: CLIPVisionModelV2 }>
  ) => {
    const { id, ipAdapterId, clipVisionModel } = action.payload;
    const ipAdapter = selectRegionalGuidanceIPAdapter(state, id, ipAdapterId);
    if (!ipAdapter) {
      return;
    }
    ipAdapter.clipVisionModel = clipVisionModel;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
