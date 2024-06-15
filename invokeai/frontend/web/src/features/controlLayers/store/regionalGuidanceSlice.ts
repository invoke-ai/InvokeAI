import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { moveOneToEnd, moveOneToStart, moveToEnd, moveToStart } from 'common/util/arrayUtils';
import { getBrushLineId, getEraserLineId, getRectShapeId } from 'features/controlLayers/konva/naming';
import type { CLIPVisionModelV2, IPMethodV2 } from 'features/controlLayers/store/types';
import { imageDTOToImageWithDims } from 'features/controlLayers/store/types';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { ParameterAutoNegative } from 'features/parameters/types/parameterSchemas';
import type { IRect } from 'konva/lib/types';
import { isEqual } from 'lodash-es';
import type { ImageDTO, IPAdapterModelConfig } from 'services/api/types';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

import type {
  BrushLineAddedArg,
  EraserLineAddedArg,
  IPAdapterData,
  PointAddedToLineArg,
  RectShapeAddedArg,
  RegionalGuidanceData,
  RgbColor,
} from './types';
import { isLine } from './types';

type RegionalGuidanceState = {
  _version: 1;
  regions: RegionalGuidanceData[];
  opacity: number;
};

const initialState: RegionalGuidanceState = {
  _version: 1,
  regions: [],
  opacity: 0.3,
};

export const selectRG = (state: RegionalGuidanceState, id: string) => state.regions.find((rg) => rg.id === id);
export const selectRGOrThrow = (state: RegionalGuidanceState, id: string) => {
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

const getRGMaskFill = (state: RegionalGuidanceState): RgbColor => {
  const lastFill = state.regions.slice(-1)[0]?.fill;
  let i = DEFAULT_MASK_COLORS.findIndex((c) => isEqual(c, lastFill));
  if (i === -1) {
    i = 0;
  }
  i = (i + 1) % DEFAULT_MASK_COLORS.length;
  const fill = DEFAULT_MASK_COLORS[i];
  assert(fill, 'This should never happen');
  return fill;
};

export const regionalGuidanceSlice = createSlice({
  name: 'regionalGuidance',
  initialState,
  reducers: {
    rgAdded: {
      reducer: (state, action: PayloadAction<{ id: string }>) => {
        const { id } = action.payload;
        const rg: RegionalGuidanceData = {
          id,
          type: 'regional_guidance',
          isEnabled: true,
          bbox: null,
          bboxNeedsUpdate: false,
          objects: [],
          fill: getRGMaskFill(state),
          x: 0,
          y: 0,
          autoNegative: 'invert',
          positivePrompt: '',
          negativePrompt: null,
          ipAdapters: [],
          imageCache: null,
        };
        state.regions.push(rg);
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
    rgRecalled: (state, action: PayloadAction<{ data: RegionalGuidanceData }>) => {
      const { data } = action.payload;
      state.regions.push(data);
    },
    rgIsEnabledToggled: (state, action: PayloadAction<{ id: string }>) => {
      const { id } = action.payload;
      const rg = selectRG(state, id);
      if (rg) {
        rg.isEnabled = !rg.isEnabled;
      }
    },
    rgTranslated: (state, action: PayloadAction<{ id: string; x: number; y: number }>) => {
      const { id, x, y } = action.payload;
      const rg = selectRG(state, id);
      if (rg) {
        rg.x = x;
        rg.y = y;
      }
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
      state.regions = state.regions.filter((ca) => ca.id !== id);
    },
    rgGlobalOpacityChanged: (state, action: PayloadAction<{ opacity: number }>) => {
      const { opacity } = action.payload;
      state.opacity = opacity;
    },
    rgMovedForwardOne: (state, action: PayloadAction<{ id: string }>) => {
      const { id } = action.payload;
      const rg = selectRG(state, id);
      if (!rg) {
        return;
      }
      moveOneToEnd(state.regions, rg);
    },
    rgMovedToFront: (state, action: PayloadAction<{ id: string }>) => {
      const { id } = action.payload;
      const rg = selectRG(state, id);
      if (!rg) {
        return;
      }
      moveToEnd(state.regions, rg);
    },
    rgMovedBackwardOne: (state, action: PayloadAction<{ id: string }>) => {
      const { id } = action.payload;
      const rg = selectRG(state, id);
      if (!rg) {
        return;
      }
      moveOneToStart(state.regions, rg);
    },
    rgMovedToBack: (state, action: PayloadAction<{ id: string }>) => {
      const { id } = action.payload;
      const rg = selectRG(state, id);
      if (!rg) {
        return;
      }
      moveToStart(state.regions, rg);
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
    rgMaskImageUploaded: (state, action: PayloadAction<{ id: string; imageDTO: ImageDTO }>) => {
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
    rgIPAdapterAdded: (state, action: PayloadAction<{ id: string; ipAdapter: IPAdapterData }>) => {
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
    rgIPAdapterImageChanged: (
      state,
      action: PayloadAction<{ id: string; ipAdapterId: string; imageDTO: ImageDTO | null }>
    ) => {
      const { id, ipAdapterId, imageDTO } = action.payload;
      const rg = selectRG(state, id);
      if (!rg) {
        return;
      }
      const ipa = rg.ipAdapters.find((ipa) => ipa.id === ipAdapterId);
      if (!ipa) {
        return;
      }
      ipa.image = imageDTO ? imageDTOToImageWithDims(imageDTO) : null;
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
    rgIPAdapterMethodChanged: (
      state,
      action: PayloadAction<{ id: string; ipAdapterId: string; method: IPMethodV2 }>
    ) => {
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
    rgBrushLineAdded: {
      reducer: (state, action: PayloadAction<BrushLineAddedArg & { lineId: string }>) => {
        const { id, points, lineId, color, width } = action.payload;
        const rg = selectRG(state, id);
        if (!rg) {
          return;
        }
        rg.objects.push({
          id: getBrushLineId(id, lineId),
          type: 'brush_line',
          points,
          strokeWidth: width,
          color,
        });
        rg.bboxNeedsUpdate = true;
        rg.imageCache = null;
      },
      prepare: (payload: BrushLineAddedArg) => ({
        payload: { ...payload, lineId: uuidv4() },
      }),
    },
    rgEraserLineAdded: {
      reducer: (state, action: PayloadAction<EraserLineAddedArg & { lineId: string }>) => {
        const { id, points, lineId, width } = action.payload;
        const rg = selectRG(state, id);
        if (!rg) {
          return;
        }
        rg.objects.push({
          id: getEraserLineId(id, lineId),
          type: 'eraser_line',
          points,
          strokeWidth: width,
        });
        rg.bboxNeedsUpdate = true;
        rg.imageCache = null;
      },
      prepare: (payload: EraserLineAddedArg) => ({
        payload: { ...payload, lineId: uuidv4() },
      }),
    },
    rgLinePointAdded: (state, action: PayloadAction<PointAddedToLineArg>) => {
      const { id, point } = action.payload;
      const rg = selectRG(state, id);
      if (!rg) {
        return;
      }
      const lastObject = rg.objects[rg.objects.length - 1];
      if (!lastObject || !isLine(lastObject)) {
        return;
      }
      lastObject.points.push(...point);
      rg.bboxNeedsUpdate = true;
      rg.imageCache = null;
    },
    rgRectAdded: {
      reducer: (state, action: PayloadAction<RectShapeAddedArg & { rectId: string }>) => {
        const { id, rect, rectId, color } = action.payload;
        if (rect.height === 0 || rect.width === 0) {
          // Ignore zero-area rectangles
          return;
        }
        const rg = selectRG(state, id);
        if (!rg) {
          return;
        }
        rg.objects.push({
          type: 'rect_shape',
          id: getRectShapeId(id, rectId),
          ...rect,
          color,
        });
        rg.bboxNeedsUpdate = true;
        rg.imageCache = null;
      },
      prepare: (payload: RectShapeAddedArg) => ({ payload: { ...payload, rectId: uuidv4() } }),
    },
    rgAllDeleted: (state) => {
      state.regions = [];
    },
  },
});

export const {
  rgAdded,
  rgRecalled,
  rgReset,
  rgIsEnabledToggled,
  rgTranslated,
  rgBboxChanged,
  rgDeleted,
  rgGlobalOpacityChanged,
  rgMovedForwardOne,
  rgMovedToFront,
  rgMovedBackwardOne,
  rgMovedToBack,
  rgPositivePromptChanged,
  rgNegativePromptChanged,
  rgFillChanged,
  rgMaskImageUploaded,
  rgAutoNegativeChanged,
  rgIPAdapterAdded,
  rgIPAdapterDeleted,
  rgIPAdapterImageChanged,
  rgIPAdapterWeightChanged,
  rgIPAdapterBeginEndStepPctChanged,
  rgIPAdapterMethodChanged,
  rgIPAdapterModelChanged,
  rgIPAdapterCLIPVisionModelChanged,
  rgBrushLineAdded,
  rgEraserLineAdded,
  rgLinePointAdded,
  rgRectAdded,
  rgAllDeleted,
} = regionalGuidanceSlice.actions;

export const selectRegionalGuidanceSlice = (state: RootState) => state.regionalGuidance;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
  return state;
};

export const regionalGuidancePersistConfig: PersistConfig<RegionalGuidanceState> = {
  name: regionalGuidanceSlice.name,
  initialState,
  migrate,
  persistDenylist: [],
};
