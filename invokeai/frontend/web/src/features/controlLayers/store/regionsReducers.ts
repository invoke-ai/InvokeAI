import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import { deepClone } from 'common/util/deepClone';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectEntity, selectRegionalGuidanceIPAdapter } from 'features/controlLayers/store/selectors';
import type {
  CanvasState,
  CLIPVisionModelV2,
  EntityIdentifierPayload,
  FillStyle,
  IPMethodV2,
  RegionalGuidanceIPAdapterConfig,
  RgbColor,
} from 'features/controlLayers/store/types';
import { getEntityIdentifier, imageDTOToImageWithDims, initialIPAdapter } from 'features/controlLayers/store/types';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { isEqual, merge } from 'lodash-es';
import type { ImageDTO, IPAdapterModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

import type { CanvasRegionalGuidanceState } from './types';

const DEFAULT_MASK_COLORS: RgbColor[] = [
  { r: 121, g: 157, b: 219 }, // rgb(121, 157, 219)
  { r: 131, g: 214, b: 131 }, // rgb(131, 214, 131)
  { r: 250, g: 225, b: 80 }, // rgb(250, 225, 80)
  { r: 220, g: 144, b: 101 }, // rgb(220, 144, 101)
  { r: 224, g: 117, b: 117 }, // rgb(224, 117, 117)
  { r: 213, g: 139, b: 202 }, // rgb(213, 139, 202)
  { r: 161, g: 120, b: 214 }, // rgb(161, 120, 214)
];

const getRGMaskFill = (state: CanvasState): RgbColor => {
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
    reducer: (
      state,
      action: PayloadAction<{ id: string; overrides?: Partial<CanvasRegionalGuidanceState>; isSelected?: boolean }>
    ) => {
      const { id, overrides, isSelected } = action.payload;
      const entity: CanvasRegionalGuidanceState = {
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
        autoNegative: true,
        positivePrompt: '',
        negativePrompt: null,
        ipAdapters: [],
      };
      merge(entity, overrides);
      state.regions.entities.push(entity);
      if (isSelected) {
        state.selectedEntityIdentifier = getEntityIdentifier(entity);
      }
    },
    prepare: (payload?: { overrides?: Partial<CanvasRegionalGuidanceState>; isSelected?: boolean }) => ({
      payload: { ...payload, id: getPrefixedId('regional_guidance') },
    }),
  },
  rgRecalled: (state, action: PayloadAction<{ data: CanvasRegionalGuidanceState }>) => {
    const { data } = action.payload;
    state.regions.entities.push(data);
    state.selectedEntityIdentifier = { type: 'regional_guidance', id: data.id };
  },
  rgPositivePromptChanged: (
    state,
    action: PayloadAction<EntityIdentifierPayload<{ prompt: string | null }, 'regional_guidance'>>
  ) => {
    const { entityIdentifier, prompt } = action.payload;
    const entity = selectEntity(state, entityIdentifier);
    if (!entity) {
      return;
    }
    entity.positivePrompt = prompt;
  },
  rgNegativePromptChanged: (
    state,
    action: PayloadAction<EntityIdentifierPayload<{ prompt: string | null }, 'regional_guidance'>>
  ) => {
    const { entityIdentifier, prompt } = action.payload;
    const entity = selectEntity(state, entityIdentifier);
    if (!entity) {
      return;
    }
    entity.negativePrompt = prompt;
  },
  rgFillColorChanged: (
    state,
    action: PayloadAction<EntityIdentifierPayload<{ color: RgbColor }, 'regional_guidance'>>
  ) => {
    const { entityIdentifier, color } = action.payload;
    const entity = selectEntity(state, entityIdentifier);
    if (!entity) {
      return;
    }
    entity.fill.color = color;
  },
  rgFillStyleChanged: (
    state,
    action: PayloadAction<EntityIdentifierPayload<{ style: FillStyle }, 'regional_guidance'>>
  ) => {
    const { entityIdentifier, style } = action.payload;
    const entity = selectEntity(state, entityIdentifier);
    if (!entity) {
      return;
    }
    entity.fill.style = style;
  },

  rgAutoNegativeToggled: (state, action: PayloadAction<EntityIdentifierPayload<void, 'regional_guidance'>>) => {
    const { entityIdentifier } = action.payload;
    const rg = selectEntity(state, entityIdentifier);
    if (!rg) {
      return;
    }
    rg.autoNegative = !rg.autoNegative;
  },
  rgIPAdapterAdded: {
    reducer: (
      state,
      action: PayloadAction<
        EntityIdentifierPayload<
          { ipAdapterId: string; overrides?: Partial<RegionalGuidanceIPAdapterConfig> },
          'regional_guidance'
        >
      >
    ) => {
      const { entityIdentifier, overrides, ipAdapterId } = action.payload;
      const entity = selectEntity(state, entityIdentifier);
      if (!entity) {
        return;
      }
      const ipAdapter = { ...deepClone(initialIPAdapter), id: ipAdapterId };
      merge(ipAdapter, overrides);
      entity.ipAdapters.push(ipAdapter);
    },
    prepare: (
      payload: EntityIdentifierPayload<{ overrides?: Partial<RegionalGuidanceIPAdapterConfig> }, 'regional_guidance'>
    ) => ({
      payload: { ...payload, ipAdapterId: getPrefixedId('regional_guidance_ip_adapter') },
    }),
  },
  rgIPAdapterDeleted: (
    state,
    action: PayloadAction<EntityIdentifierPayload<{ ipAdapterId: string }, 'regional_guidance'>>
  ) => {
    const { entityIdentifier, ipAdapterId } = action.payload;
    const entity = selectEntity(state, entityIdentifier);
    if (!entity) {
      return;
    }
    entity.ipAdapters = entity.ipAdapters.filter((ipAdapter) => ipAdapter.id !== ipAdapterId);
  },
  rgIPAdapterImageChanged: (
    state,
    action: PayloadAction<
      EntityIdentifierPayload<{ ipAdapterId: string; imageDTO: ImageDTO | null }, 'regional_guidance'>
    >
  ) => {
    const { entityIdentifier, ipAdapterId, imageDTO } = action.payload;
    const ipAdapter = selectRegionalGuidanceIPAdapter(state, entityIdentifier, ipAdapterId);
    if (!ipAdapter) {
      return;
    }
    ipAdapter.image = imageDTO ? imageDTOToImageWithDims(imageDTO) : null;
  },
  rgIPAdapterWeightChanged: (
    state,
    action: PayloadAction<EntityIdentifierPayload<{ ipAdapterId: string; weight: number }, 'regional_guidance'>>
  ) => {
    const { entityIdentifier, ipAdapterId, weight } = action.payload;
    const ipAdapter = selectRegionalGuidanceIPAdapter(state, entityIdentifier, ipAdapterId);
    if (!ipAdapter) {
      return;
    }
    ipAdapter.weight = weight;
  },
  rgIPAdapterBeginEndStepPctChanged: (
    state,
    action: PayloadAction<
      EntityIdentifierPayload<{ ipAdapterId: string; beginEndStepPct: [number, number] }, 'regional_guidance'>
    >
  ) => {
    const { entityIdentifier, ipAdapterId, beginEndStepPct } = action.payload;
    const ipAdapter = selectRegionalGuidanceIPAdapter(state, entityIdentifier, ipAdapterId);
    if (!ipAdapter) {
      return;
    }
    ipAdapter.beginEndStepPct = beginEndStepPct;
  },
  rgIPAdapterMethodChanged: (
    state,
    action: PayloadAction<EntityIdentifierPayload<{ ipAdapterId: string; method: IPMethodV2 }, 'regional_guidance'>>
  ) => {
    const { entityIdentifier, ipAdapterId, method } = action.payload;
    const ipAdapter = selectRegionalGuidanceIPAdapter(state, entityIdentifier, ipAdapterId);
    if (!ipAdapter) {
      return;
    }
    ipAdapter.method = method;
  },
  rgIPAdapterModelChanged: (
    state,
    action: PayloadAction<
      EntityIdentifierPayload<
        {
          ipAdapterId: string;
          modelConfig: IPAdapterModelConfig | null;
        },
        'regional_guidance'
      >
    >
  ) => {
    const { entityIdentifier, ipAdapterId, modelConfig } = action.payload;
    const ipAdapter = selectRegionalGuidanceIPAdapter(state, entityIdentifier, ipAdapterId);
    if (!ipAdapter) {
      return;
    }
    ipAdapter.model = modelConfig ? zModelIdentifierField.parse(modelConfig) : null;
  },
  rgIPAdapterCLIPVisionModelChanged: (
    state,
    action: PayloadAction<
      EntityIdentifierPayload<{ ipAdapterId: string; clipVisionModel: CLIPVisionModelV2 }, 'regional_guidance'>
    >
  ) => {
    const { entityIdentifier, ipAdapterId, clipVisionModel } = action.payload;
    const ipAdapter = selectRegionalGuidanceIPAdapter(state, entityIdentifier, ipAdapterId);
    if (!ipAdapter) {
      return;
    }
    ipAdapter.clipVisionModel = clipVisionModel;
  },
} satisfies SliceCaseReducers<CanvasState>;
