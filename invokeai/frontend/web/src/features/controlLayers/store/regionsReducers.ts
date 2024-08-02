import type { PayloadAction, SliceCaseReducers } from '@reduxjs/toolkit';
import { moveOneToEnd, moveOneToStart, moveToEnd, moveToStart } from 'common/util/arrayUtils';
import type {
  CanvasBrushLineState,
  CanvasV2State,
  CLIPVisionModelV2,
  CanvasEraserLineState,
  IPMethodV2,
  PositionChangedArg,
  CanvasRectState,
  ScaleChangedArg,
} from 'features/controlLayers/store/types';
import { imageDTOToImageObject, imageDTOToImageWithDims } from 'features/controlLayers/store/types';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { ParameterAutoNegative } from 'features/parameters/types/parameterSchemas';
import type { IRect } from 'konva/lib/types';
import { isEqual } from 'lodash-es';
import type { ImageDTO, IPAdapterModelConfig } from 'services/api/types';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

import type { CanvasIPAdapterState, CanvasRegionalGuidanceState, RgbColor } from './types';

export const selectRG = (state: CanvasV2State, id: string) => state.regions.entities.find((rg) => rg.id === id);
export const selectRGOrThrow = (state: CanvasV2State, id: string) => {
  const rg = selectRG(state, id);
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
  const lastFill = state.regions.entities.slice(-1)[0]?.fill;
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
        type: 'regional_guidance',
        isEnabled: true,
        bbox: null,
        bboxNeedsUpdate: false,
        objects: [],
        fill: getRGMaskFill(state),
        position: { x: 0, y: 0 },
        autoNegative: 'invert',
        positivePrompt: '',
        negativePrompt: null,
        ipAdapters: [],
        imageCache: null,
      };
      state.regions.entities.push(rg);
      state.selectedEntityIdentifier = { type: 'regional_guidance', id };
    },
    prepare: () => ({ payload: { id: uuidv4() } }),
  },
  rgReset: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const rg = selectRG(state, id);
    if (!rg) {
      return;
    }
    rg.objects = [];
    rg.bbox = null;
    rg.bboxNeedsUpdate = false;
    rg.imageCache = null;
  },
  rgRecalled: (state, action: PayloadAction<{ data: CanvasRegionalGuidanceState }>) => {
    const { data } = action.payload;
    state.regions.entities.push(data);
    state.selectedEntityIdentifier = { type: 'regional_guidance', id: data.id };
  },
  rgIsEnabledToggled: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const rg = selectRG(state, id);
    if (rg) {
      rg.isEnabled = !rg.isEnabled;
    }
  },
  rgTranslated: (state, action: PayloadAction<PositionChangedArg>) => {
    const { id, position } = action.payload;
    const rg = selectRG(state, id);
    if (rg) {
      rg.position = position;
    }
  },
  rgScaled: (state, action: PayloadAction<ScaleChangedArg>) => {
    const { id, scale, position } = action.payload;
    const rg = selectRG(state, id);
    if (!rg) {
      return;
    }
    for (const obj of rg.objects) {
      if (obj.type === 'brush_line') {
        obj.points = obj.points.map((point) => point * scale);
        obj.strokeWidth *= scale;
      } else if (obj.type === 'eraser_line') {
        obj.points = obj.points.map((point) => point * scale);
        obj.strokeWidth *= scale;
      } else if (obj.type === 'rect') {
        obj.x *= scale;
        obj.y *= scale;
        obj.height *= scale;
        obj.width *= scale;
      }
    }
    rg.position = position;
    rg.bboxNeedsUpdate = true;
    state.layers.imageCache = null;
  },
  rgBboxChanged: (state, action: PayloadAction<{ id: string; bbox: IRect | null }>) => {
    const { id, bbox } = action.payload;
    const rg = selectRG(state, id);
    if (rg) {
      rg.bbox = bbox;
      rg.bboxNeedsUpdate = false;
    }
  },
  rgDeleted: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    state.regions.entities = state.regions.entities.filter((ca) => ca.id !== id);
  },
  rgAllDeleted: (state) => {
    state.regions.entities = [];
  },
  rgMovedForwardOne: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const rg = selectRG(state, id);
    if (!rg) {
      return;
    }
    moveOneToEnd(state.regions.entities, rg);
  },
  rgMovedToFront: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const rg = selectRG(state, id);
    if (!rg) {
      return;
    }
    moveToEnd(state.regions.entities, rg);
  },
  rgMovedBackwardOne: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const rg = selectRG(state, id);
    if (!rg) {
      return;
    }
    moveOneToStart(state.regions.entities, rg);
  },
  rgMovedToBack: (state, action: PayloadAction<{ id: string }>) => {
    const { id } = action.payload;
    const rg = selectRG(state, id);
    if (!rg) {
      return;
    }
    moveToStart(state.regions.entities, rg);
  },
  rgPositivePromptChanged: (state, action: PayloadAction<{ id: string; prompt: string | null }>) => {
    const { id, prompt } = action.payload;
    const rg = selectRG(state, id);
    if (!rg) {
      return;
    }
    rg.positivePrompt = prompt;
  },
  rgNegativePromptChanged: (state, action: PayloadAction<{ id: string; prompt: string | null }>) => {
    const { id, prompt } = action.payload;
    const rg = selectRG(state, id);
    if (!rg) {
      return;
    }
    rg.negativePrompt = prompt;
  },
  rgFillChanged: (state, action: PayloadAction<{ id: string; fill: RgbColor }>) => {
    const { id, fill } = action.payload;
    const rg = selectRG(state, id);
    if (!rg) {
      return;
    }
    rg.fill = fill;
  },
  rgImageCacheChanged: (state, action: PayloadAction<{ id: string; imageDTO: ImageDTO }>) => {
    const { id, imageDTO } = action.payload;
    const rg = selectRG(state, id);
    if (!rg) {
      return;
    }
    rg.imageCache = imageDTOToImageWithDims(imageDTO);
  },
  rgAutoNegativeChanged: (state, action: PayloadAction<{ id: string; autoNegative: ParameterAutoNegative }>) => {
    const { id, autoNegative } = action.payload;
    const rg = selectRG(state, id);
    if (!rg) {
      return;
    }
    rg.autoNegative = autoNegative;
  },
  rgIPAdapterAdded: (state, action: PayloadAction<{ id: string; ipAdapter: CanvasIPAdapterState }>) => {
    const { id, ipAdapter } = action.payload;
    const rg = selectRG(state, id);
    if (!rg) {
      return;
    }
    rg.ipAdapters.push(ipAdapter);
  },
  rgIPAdapterDeleted: (state, action: PayloadAction<{ id: string; ipAdapterId: string }>) => {
    const { id, ipAdapterId } = action.payload;
    const rg = selectRG(state, id);
    if (!rg) {
      return;
    }
    rg.ipAdapters = rg.ipAdapters.filter((ipAdapter) => ipAdapter.id !== ipAdapterId);
  },
  rgIPAdapterImageChanged: {
    reducer: (
      state,
      action: PayloadAction<{ id: string; ipAdapterId: string; imageDTO: ImageDTO | null; objectId: string }>
    ) => {
      const { id, ipAdapterId, imageDTO, objectId } = action.payload;
      const rg = selectRG(state, id);
      if (!rg) {
        return;
      }
      const ipa = rg.ipAdapters.find((ipa) => ipa.id === ipAdapterId);
      if (!ipa) {
        return;
      }
      ipa.imageObject = imageDTO ? imageDTOToImageObject(id, objectId, imageDTO) : null;
    },
    prepare: (payload: { id: string; ipAdapterId: string; imageDTO: ImageDTO | null }) => ({
      payload: { ...payload, objectId: uuidv4() },
    }),
  },
  rgIPAdapterWeightChanged: (state, action: PayloadAction<{ id: string; ipAdapterId: string; weight: number }>) => {
    const { id, ipAdapterId, weight } = action.payload;
    const rg = selectRG(state, id);
    if (!rg) {
      return;
    }
    const ipa = rg.ipAdapters.find((ipa) => ipa.id === ipAdapterId);
    if (!ipa) {
      return;
    }
    ipa.weight = weight;
  },
  rgIPAdapterBeginEndStepPctChanged: (
    state,
    action: PayloadAction<{ id: string; ipAdapterId: string; beginEndStepPct: [number, number] }>
  ) => {
    const { id, ipAdapterId, beginEndStepPct } = action.payload;
    const rg = selectRG(state, id);
    if (!rg) {
      return;
    }
    const ipa = rg.ipAdapters.find((ipa) => ipa.id === ipAdapterId);
    if (!ipa) {
      return;
    }
    ipa.beginEndStepPct = beginEndStepPct;
  },
  rgIPAdapterMethodChanged: (state, action: PayloadAction<{ id: string; ipAdapterId: string; method: IPMethodV2 }>) => {
    const { id, ipAdapterId, method } = action.payload;
    const rg = selectRG(state, id);
    if (!rg) {
      return;
    }
    const ipa = rg.ipAdapters.find((ipa) => ipa.id === ipAdapterId);
    if (!ipa) {
      return;
    }
    ipa.method = method;
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
    const rg = selectRG(state, id);
    if (!rg) {
      return;
    }
    const ipa = rg.ipAdapters.find((ipa) => ipa.id === ipAdapterId);
    if (!ipa) {
      return;
    }
    if (modelConfig) {
      ipa.model = zModelIdentifierField.parse(modelConfig);
    } else {
      ipa.model = null;
    }
  },
  rgIPAdapterCLIPVisionModelChanged: (
    state,
    action: PayloadAction<{ id: string; ipAdapterId: string; clipVisionModel: CLIPVisionModelV2 }>
  ) => {
    const { id, ipAdapterId, clipVisionModel } = action.payload;
    const rg = selectRG(state, id);
    if (!rg) {
      return;
    }
    const ipa = rg.ipAdapters.find((ipa) => ipa.id === ipAdapterId);
    if (!ipa) {
      return;
    }
    ipa.clipVisionModel = clipVisionModel;
  },
  rgBrushLineAdded: (state, action: PayloadAction<{ id: string; brushLine: CanvasBrushLineState }>) => {
    const { id, brushLine } = action.payload;
    const rg = selectRG(state, id);
    if (!rg) {
      return;
    }

    rg.objects.push(brushLine);
    rg.bboxNeedsUpdate = true;
    state.layers.imageCache = null;
  },
  rgEraserLineAdded: (state, action: PayloadAction<{ id: string; eraserLine: CanvasEraserLineState }>) => {
    const { id, eraserLine } = action.payload;
    const rg = selectRG(state, id);
    if (!rg) {
      return;
    }

    rg.objects.push(eraserLine);
    rg.bboxNeedsUpdate = true;
    state.layers.imageCache = null;
  },
  rgRectShapeAdded: (state, action: PayloadAction<{ id: string; rectShape: CanvasRectState }>) => {
    const { id, rectShape } = action.payload;
    const rg = selectRG(state, id);
    if (!rg) {
      return;
    }

    rg.objects.push(rectShape);
    rg.bboxNeedsUpdate = true;
    state.layers.imageCache = null;
  },
} satisfies SliceCaseReducers<CanvasV2State>;
