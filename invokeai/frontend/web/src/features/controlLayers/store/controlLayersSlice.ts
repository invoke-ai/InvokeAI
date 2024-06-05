import type { PayloadAction, UnknownAction } from '@reduxjs/toolkit';
import { createSlice, isAnyOf } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { moveBackward, moveForward, moveToBack, moveToFront } from 'common/util/arrayUtils';
import { deepClone } from 'common/util/deepClone';
import { roundDownToMultiple } from 'common/util/roundDownToMultiple';
import {
  getBrushLineId,
  getCALayerId,
  getEraserLineId,
  getIPALayerId,
  getRasterLayerId,
  getRectId,
  getRGLayerId,
  INITIAL_IMAGE_LAYER_ID,
} from 'features/controlLayers/konva/naming';
import type {
  CLIPVisionModelV2,
  ControlModeV2,
  ControlNetConfigV2,
  IPAdapterConfigV2,
  IPMethodV2,
  ProcessorConfig,
  T2IAdapterConfigV2,
} from 'features/controlLayers/util/controlAdapters';
import {
  buildControlAdapterProcessorV2,
  controlNetToT2IAdapter,
  imageDTOToImageWithDims,
  t2iAdapterToControlNet,
} from 'features/controlLayers/util/controlAdapters';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { calculateNewSize } from 'features/parameters/components/ImageSize/calculateNewSize';
import { initialAspectRatioState } from 'features/parameters/components/ImageSize/constants';
import type { AspectRatioState } from 'features/parameters/components/ImageSize/types';
import { modelChanged } from 'features/parameters/store/generationSlice';
import type { ParameterAutoNegative } from 'features/parameters/types/parameterSchemas';
import { getIsSizeOptimal, getOptimalDimension } from 'features/parameters/util/optimalDimension';
import type { IRect, Vector2d } from 'konva/lib/types';
import { isEqual, partition, unset } from 'lodash-es';
import { atom } from 'nanostores';
import type { RgbColor } from 'react-colorful';
import type { UndoableOptions } from 'redux-undo';
import type { ControlNetModelConfig, ImageDTO, IPAdapterModelConfig, T2IAdapterModelConfig } from 'services/api/types';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

import type {
  AddBrushLineArg,
  AddEraserLineArg,
  AddPointToLineArg,
  AddRectShapeArg,
  ControlAdapterLayer,
  ControlLayersState,
  InitialImageLayer,
  IPAdapterLayer,
  Layer,
  RasterLayer,
  RegionalGuidanceLayer,
  RgbaColor,
  Tool,
} from './types';
import {
  DEFAULT_RGBA_COLOR,
  isCAOrIPALayer,
  isControlAdapterLayer,
  isInitialImageLayer,
  isIPAdapterLayer,
  isLine,
  isRasterLayer,
  isRegionalGuidanceLayer,
  isRenderableLayer,
  isRGOrRasterlayer,
} from './types';

export const initialControlLayersState: ControlLayersState = {
  _version: 3,
  selectedLayerId: null,
  brushSize: 100,
  brushColor: DEFAULT_RGBA_COLOR,
  layers: [],
  globalMaskLayerOpacity: 0.3, // this globally changes all mask layers' opacity
  positivePrompt: '',
  negativePrompt: '',
  positivePrompt2: '',
  negativePrompt2: '',
  shouldConcatPrompts: true,
  size: {
    width: 512,
    height: 512,
    aspectRatio: deepClone(initialAspectRatioState),
  },
};

/**
 * A selector that accepts a type guard and returns the first layer that matches the guard.
 * Throws if the layer is not found or does not match the guard.
 */
export const selectLayerOrThrow = <T extends Layer>(
  state: ControlLayersState,
  layerId: string,
  predicate: (layer: Layer) => layer is T
): T => {
  const layer = state.layers.find((l) => l.id === layerId);
  assert(layer && predicate(layer));
  return layer;
};

export const selectRGLayerIPAdapterOrThrow = (
  state: ControlLayersState,
  layerId: string,
  ipAdapterId: string
): IPAdapterConfigV2 => {
  const layer = selectLayerOrThrow(state, layerId, isRegionalGuidanceLayer);
  const ipAdapter = layer.ipAdapters.find((ipAdapter) => ipAdapter.id === ipAdapterId);
  assert(ipAdapter);
  return ipAdapter;
};

const getVectorMaskPreviewColor = (state: ControlLayersState): RgbColor => {
  const rgLayers = state.layers.filter(isRegionalGuidanceLayer);
  const lastColor = rgLayers[rgLayers.length - 1]?.previewColor;
  return LayerColors.next(lastColor);
};
const exclusivelySelectLayer = (state: ControlLayersState, layerId: string) => {
  for (const layer of state.layers) {
    layer.isSelected = layer.id === layerId;
  }
  state.selectedLayerId = layerId;
};

export const controlLayersSlice = createSlice({
  name: 'controlLayers',
  initialState: initialControlLayersState,
  reducers: {
    //#region Any Layer Type
    layerSelected: (state, action: PayloadAction<string>) => {
      exclusivelySelectLayer(state, action.payload);
    },
    layerIsEnabledToggled: (state, action: PayloadAction<string>) => {
      const layer = state.layers.find((l) => l.id === action.payload);
      if (layer) {
        layer.isEnabled = !layer.isEnabled;
      }
    },
    layerTranslated: (state, action: PayloadAction<{ layerId: string; x: number; y: number }>) => {
      const { layerId, x, y } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (isRenderableLayer(layer)) {
        layer.x = x;
        layer.y = y;
      }
      if (isRegionalGuidanceLayer(layer)) {
        layer.uploadedMaskImage = null;
      }
    },
    layerBboxChanged: (state, action: PayloadAction<{ layerId: string; bbox: IRect | null }>) => {
      const { layerId, bbox } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (isRenderableLayer(layer)) {
        layer.bbox = bbox;
        layer.bboxNeedsUpdate = false;
        if (bbox === null && layer.type === 'regional_guidance_layer') {
          // The layer was fully erased, empty its objects to prevent accumulation of invisible objects
          layer.objects = [];
          layer.uploadedMaskImage = null;
        }
      }
    },
    layerReset: (state, action: PayloadAction<string>) => {
      const layer = state.layers.find((l) => l.id === action.payload);
      // TODO(psyche): Should other layer types also have reset functionality?
      if (isRegionalGuidanceLayer(layer)) {
        layer.objects = [];
        layer.bbox = null;
        layer.isEnabled = true;
        layer.bboxNeedsUpdate = false;
        layer.uploadedMaskImage = null;
      }
      if (isRasterLayer(layer)) {
        layer.isEnabled = true;
        layer.objects = [];
        layer.bbox = null;
        layer.bboxNeedsUpdate = false;
      }
    },
    layerDeleted: (state, action: PayloadAction<string>) => {
      state.layers = state.layers.filter((l) => l.id !== action.payload);
      state.selectedLayerId = state.layers[0]?.id ?? null;
    },
    layerOpacityChanged: (state, action: PayloadAction<{ layerId: string; opacity: number }>) => {
      const { layerId, opacity } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (isControlAdapterLayer(layer) || isInitialImageLayer(layer) || isRasterLayer(layer)) {
        layer.opacity = opacity;
      }
    },
    layerMovedForward: (state, action: PayloadAction<string>) => {
      const cb = (l: Layer) => l.id === action.payload;
      const [renderableLayers, ipAdapterLayers] = partition(state.layers, isRenderableLayer);
      moveForward(renderableLayers, cb);
      state.layers = [...ipAdapterLayers, ...renderableLayers];
    },
    layerMovedToFront: (state, action: PayloadAction<string>) => {
      const cb = (l: Layer) => l.id === action.payload;
      const [renderableLayers, ipAdapterLayers] = partition(state.layers, isRenderableLayer);
      // Because the layers are in reverse order, moving to the front is equivalent to moving to the back
      moveToBack(renderableLayers, cb);
      state.layers = [...ipAdapterLayers, ...renderableLayers];
    },
    layerMovedBackward: (state, action: PayloadAction<string>) => {
      const cb = (l: Layer) => l.id === action.payload;
      const [renderableLayers, ipAdapterLayers] = partition(state.layers, isRenderableLayer);
      moveBackward(renderableLayers, cb);
      state.layers = [...ipAdapterLayers, ...renderableLayers];
    },
    layerMovedToBack: (state, action: PayloadAction<string>) => {
      const cb = (l: Layer) => l.id === action.payload;
      const [renderableLayers, ipAdapterLayers] = partition(state.layers, isRenderableLayer);
      // Because the layers are in reverse order, moving to the back is equivalent to moving to the front
      moveToFront(renderableLayers, cb);
      state.layers = [...ipAdapterLayers, ...renderableLayers];
    },
    selectedLayerDeleted: (state) => {
      state.layers = state.layers.filter((l) => l.id !== state.selectedLayerId);
      state.selectedLayerId = state.layers[0]?.id ?? null;
    },
    allLayersDeleted: (state) => {
      state.layers = [];
      state.selectedLayerId = null;
    },
    //#endregion

    //#region CA Layers
    caLayerAdded: {
      reducer: (
        state,
        action: PayloadAction<{ layerId: string; controlAdapter: ControlNetConfigV2 | T2IAdapterConfigV2 }>
      ) => {
        const { layerId, controlAdapter } = action.payload;
        const layer: ControlAdapterLayer = {
          id: getCALayerId(layerId),
          type: 'control_adapter_layer',
          x: 0,
          y: 0,
          bbox: null,
          bboxNeedsUpdate: false,
          isEnabled: true,
          opacity: 1,
          isSelected: true,
          isFilterEnabled: true,
          controlAdapter,
        };
        state.layers.push(layer);
        exclusivelySelectLayer(state, layer.id);
      },
      prepare: (controlAdapter: ControlNetConfigV2 | T2IAdapterConfigV2) => ({
        payload: { layerId: uuidv4(), controlAdapter },
      }),
    },
    caLayerRecalled: (state, action: PayloadAction<ControlAdapterLayer>) => {
      state.layers.push({ ...action.payload, isSelected: true });
      exclusivelySelectLayer(state, action.payload.id);
    },
    caLayerImageChanged: (state, action: PayloadAction<{ layerId: string; imageDTO: ImageDTO | null }>) => {
      const { layerId, imageDTO } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isControlAdapterLayer);
      layer.bbox = null;
      layer.bboxNeedsUpdate = true;
      layer.isEnabled = true;
      if (imageDTO) {
        const newImage = imageDTOToImageWithDims(imageDTO);
        if (isEqual(newImage, layer.controlAdapter.image)) {
          return;
        }
        layer.controlAdapter.image = newImage;
        layer.controlAdapter.processedImage = null;
      } else {
        layer.controlAdapter.image = null;
        layer.controlAdapter.processedImage = null;
      }
    },
    caLayerProcessedImageChanged: (state, action: PayloadAction<{ layerId: string; imageDTO: ImageDTO | null }>) => {
      const { layerId, imageDTO } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isControlAdapterLayer);
      layer.bbox = null;
      layer.bboxNeedsUpdate = true;
      layer.isEnabled = true;
      layer.controlAdapter.processedImage = imageDTO ? imageDTOToImageWithDims(imageDTO) : null;
    },
    caLayerModelChanged: (
      state,
      action: PayloadAction<{
        layerId: string;
        modelConfig: ControlNetModelConfig | T2IAdapterModelConfig | null;
      }>
    ) => {
      const { layerId, modelConfig } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isControlAdapterLayer);
      if (!modelConfig) {
        layer.controlAdapter.model = null;
        return;
      }
      layer.controlAdapter.model = zModelIdentifierField.parse(modelConfig);

      // We may need to convert the CA to match the model
      if (layer.controlAdapter.type === 't2i_adapter' && layer.controlAdapter.model.type === 'controlnet') {
        layer.controlAdapter = t2iAdapterToControlNet(layer.controlAdapter);
      } else if (layer.controlAdapter.type === 'controlnet' && layer.controlAdapter.model.type === 't2i_adapter') {
        layer.controlAdapter = controlNetToT2IAdapter(layer.controlAdapter);
      }

      const candidateProcessorConfig = buildControlAdapterProcessorV2(modelConfig);
      if (candidateProcessorConfig?.type !== layer.controlAdapter.processorConfig?.type) {
        // The processor has changed. For example, the previous model was a Canny model and the new model is a Depth
        // model. We need to use the new processor.
        layer.controlAdapter.processedImage = null;
        layer.controlAdapter.processorConfig = candidateProcessorConfig;
      }
    },
    caLayerControlModeChanged: (state, action: PayloadAction<{ layerId: string; controlMode: ControlModeV2 }>) => {
      const { layerId, controlMode } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isControlAdapterLayer);
      assert(layer.controlAdapter.type === 'controlnet');
      layer.controlAdapter.controlMode = controlMode;
    },
    caLayerProcessorConfigChanged: (
      state,
      action: PayloadAction<{ layerId: string; processorConfig: ProcessorConfig | null }>
    ) => {
      const { layerId, processorConfig } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isControlAdapterLayer);
      layer.controlAdapter.processorConfig = processorConfig;
      if (!processorConfig) {
        layer.controlAdapter.processedImage = null;
      }
    },
    caLayerIsFilterEnabledChanged: (state, action: PayloadAction<{ layerId: string; isFilterEnabled: boolean }>) => {
      const { layerId, isFilterEnabled } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isControlAdapterLayer);
      layer.isFilterEnabled = isFilterEnabled;
    },
    caLayerProcessorPendingBatchIdChanged: (
      state,
      action: PayloadAction<{ layerId: string; batchId: string | null }>
    ) => {
      const { layerId, batchId } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isControlAdapterLayer);
      layer.controlAdapter.processorPendingBatchId = batchId;
    },
    //#endregion

    //#region IP Adapter Layers
    ipaLayerAdded: {
      reducer: (state, action: PayloadAction<{ layerId: string; ipAdapter: IPAdapterConfigV2 }>) => {
        const { layerId, ipAdapter } = action.payload;
        const layer: IPAdapterLayer = {
          id: getIPALayerId(layerId),
          type: 'ip_adapter_layer',
          isEnabled: true,
          isSelected: true,
          ipAdapter,
        };
        state.layers.push(layer);
        exclusivelySelectLayer(state, layer.id);
      },
      prepare: (ipAdapter: IPAdapterConfigV2) => ({ payload: { layerId: uuidv4(), ipAdapter } }),
    },
    ipaLayerRecalled: (state, action: PayloadAction<IPAdapterLayer>) => {
      state.layers.push(action.payload);
    },
    ipaLayerImageChanged: (state, action: PayloadAction<{ layerId: string; imageDTO: ImageDTO | null }>) => {
      const { layerId, imageDTO } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isIPAdapterLayer);
      layer.ipAdapter.image = imageDTO ? imageDTOToImageWithDims(imageDTO) : null;
    },
    ipaLayerMethodChanged: (state, action: PayloadAction<{ layerId: string; method: IPMethodV2 }>) => {
      const { layerId, method } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isIPAdapterLayer);
      layer.ipAdapter.method = method;
    },
    ipaLayerModelChanged: (
      state,
      action: PayloadAction<{
        layerId: string;
        modelConfig: IPAdapterModelConfig | null;
      }>
    ) => {
      const { layerId, modelConfig } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isIPAdapterLayer);
      if (!modelConfig) {
        layer.ipAdapter.model = null;
        return;
      }
      layer.ipAdapter.model = zModelIdentifierField.parse(modelConfig);
    },
    ipaLayerCLIPVisionModelChanged: (
      state,
      action: PayloadAction<{ layerId: string; clipVisionModel: CLIPVisionModelV2 }>
    ) => {
      const { layerId, clipVisionModel } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isIPAdapterLayer);
      layer.ipAdapter.clipVisionModel = clipVisionModel;
    },
    //#endregion

    //#region CA or IPA Layers
    caOrIPALayerWeightChanged: (state, action: PayloadAction<{ layerId: string; weight: number }>) => {
      const { layerId, weight } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isCAOrIPALayer);
      if (layer.type === 'control_adapter_layer') {
        layer.controlAdapter.weight = weight;
      } else {
        layer.ipAdapter.weight = weight;
      }
    },
    caOrIPALayerBeginEndStepPctChanged: (
      state,
      action: PayloadAction<{ layerId: string; beginEndStepPct: [number, number] }>
    ) => {
      const { layerId, beginEndStepPct } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isCAOrIPALayer);
      if (layer.type === 'control_adapter_layer') {
        layer.controlAdapter.beginEndStepPct = beginEndStepPct;
      } else {
        layer.ipAdapter.beginEndStepPct = beginEndStepPct;
      }
    },
    //#endregion

    //#region RG Layers
    rgLayerAdded: {
      reducer: (state, action: PayloadAction<{ layerId: string }>) => {
        const { layerId } = action.payload;
        const layer: RegionalGuidanceLayer = {
          id: getRGLayerId(layerId),
          type: 'regional_guidance_layer',
          isEnabled: true,
          bbox: null,
          bboxNeedsUpdate: false,
          objects: [],
          previewColor: getVectorMaskPreviewColor(state),
          x: 0,
          y: 0,
          autoNegative: 'invert',
          positivePrompt: '',
          negativePrompt: null,
          ipAdapters: [],
          isSelected: true,
          uploadedMaskImage: null,
        };
        state.layers.push(layer);
        exclusivelySelectLayer(state, layer.id);
      },
      prepare: () => ({ payload: { layerId: uuidv4() } }),
    },
    rgLayerRecalled: (state, action: PayloadAction<RegionalGuidanceLayer>) => {
      state.layers.push({ ...action.payload, isSelected: true });
      exclusivelySelectLayer(state, action.payload.id);
    },
    rgLayerPositivePromptChanged: (state, action: PayloadAction<{ layerId: string; prompt: string | null }>) => {
      const { layerId, prompt } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isRegionalGuidanceLayer);
      layer.positivePrompt = prompt;
    },
    rgLayerNegativePromptChanged: (state, action: PayloadAction<{ layerId: string; prompt: string | null }>) => {
      const { layerId, prompt } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isRegionalGuidanceLayer);
      layer.negativePrompt = prompt;
    },
    rgLayerPreviewColorChanged: (state, action: PayloadAction<{ layerId: string; color: RgbColor }>) => {
      const { layerId, color } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isRegionalGuidanceLayer);
      layer.previewColor = color;
    },

    rgLayerMaskImageUploaded: (state, action: PayloadAction<{ layerId: string; imageDTO: ImageDTO }>) => {
      const { layerId, imageDTO } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isRegionalGuidanceLayer);
      layer.uploadedMaskImage = imageDTOToImageWithDims(imageDTO);
    },
    rgLayerAutoNegativeChanged: (
      state,
      action: PayloadAction<{ layerId: string; autoNegative: ParameterAutoNegative }>
    ) => {
      const { layerId, autoNegative } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isRegionalGuidanceLayer);
      layer.autoNegative = autoNegative;
    },
    rgLayerIPAdapterAdded: (state, action: PayloadAction<{ layerId: string; ipAdapter: IPAdapterConfigV2 }>) => {
      const { layerId, ipAdapter } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isRegionalGuidanceLayer);
      layer.ipAdapters.push(ipAdapter);
    },
    rgLayerIPAdapterDeleted: (state, action: PayloadAction<{ layerId: string; ipAdapterId: string }>) => {
      const { layerId, ipAdapterId } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isRegionalGuidanceLayer);
      layer.ipAdapters = layer.ipAdapters.filter((ipAdapter) => ipAdapter.id !== ipAdapterId);
    },
    rgLayerIPAdapterImageChanged: (
      state,
      action: PayloadAction<{ layerId: string; ipAdapterId: string; imageDTO: ImageDTO | null }>
    ) => {
      const { layerId, ipAdapterId, imageDTO } = action.payload;
      const ipAdapter = selectRGLayerIPAdapterOrThrow(state, layerId, ipAdapterId);
      ipAdapter.image = imageDTO ? imageDTOToImageWithDims(imageDTO) : null;
    },
    rgLayerIPAdapterWeightChanged: (
      state,
      action: PayloadAction<{ layerId: string; ipAdapterId: string; weight: number }>
    ) => {
      const { layerId, ipAdapterId, weight } = action.payload;
      const ipAdapter = selectRGLayerIPAdapterOrThrow(state, layerId, ipAdapterId);
      ipAdapter.weight = weight;
    },
    rgLayerIPAdapterBeginEndStepPctChanged: (
      state,
      action: PayloadAction<{ layerId: string; ipAdapterId: string; beginEndStepPct: [number, number] }>
    ) => {
      const { layerId, ipAdapterId, beginEndStepPct } = action.payload;
      const ipAdapter = selectRGLayerIPAdapterOrThrow(state, layerId, ipAdapterId);
      ipAdapter.beginEndStepPct = beginEndStepPct;
    },
    rgLayerIPAdapterMethodChanged: (
      state,
      action: PayloadAction<{ layerId: string; ipAdapterId: string; method: IPMethodV2 }>
    ) => {
      const { layerId, ipAdapterId, method } = action.payload;
      const ipAdapter = selectRGLayerIPAdapterOrThrow(state, layerId, ipAdapterId);
      ipAdapter.method = method;
    },
    rgLayerIPAdapterModelChanged: (
      state,
      action: PayloadAction<{
        layerId: string;
        ipAdapterId: string;
        modelConfig: IPAdapterModelConfig | null;
      }>
    ) => {
      const { layerId, ipAdapterId, modelConfig } = action.payload;
      const ipAdapter = selectRGLayerIPAdapterOrThrow(state, layerId, ipAdapterId);
      if (!modelConfig) {
        ipAdapter.model = null;
        return;
      }
      ipAdapter.model = zModelIdentifierField.parse(modelConfig);
    },
    rgLayerIPAdapterCLIPVisionModelChanged: (
      state,
      action: PayloadAction<{ layerId: string; ipAdapterId: string; clipVisionModel: CLIPVisionModelV2 }>
    ) => {
      const { layerId, ipAdapterId, clipVisionModel } = action.payload;
      const ipAdapter = selectRGLayerIPAdapterOrThrow(state, layerId, ipAdapterId);
      ipAdapter.clipVisionModel = clipVisionModel;
    },
    //#endregion

    //#region Initial Image Layer
    iiLayerAdded: {
      reducer: (state, action: PayloadAction<{ layerId: string; imageDTO: ImageDTO | null }>) => {
        const { layerId, imageDTO } = action.payload;

        // Retain opacity and denoising strength of existing initial image layer if exists
        let opacity = 1;
        let denoisingStrength = 0.75;
        const iiLayer = state.layers.find((l) => l.id === layerId);
        if (iiLayer) {
          assert(isInitialImageLayer(iiLayer));
          opacity = iiLayer.opacity;
          denoisingStrength = iiLayer.denoisingStrength;
        }

        // Highlander! There can be only one!
        state.layers = state.layers.filter((l) => (isInitialImageLayer(l) ? false : true));

        const layer: InitialImageLayer = {
          id: layerId,
          type: 'initial_image_layer',
          opacity,
          x: 0,
          y: 0,
          bbox: null,
          bboxNeedsUpdate: false,
          isEnabled: true,
          image: imageDTO ? imageDTOToImageWithDims(imageDTO) : null,
          isSelected: true,
          denoisingStrength,
        };
        state.layers.push(layer);
        exclusivelySelectLayer(state, layer.id);
      },
      prepare: (imageDTO: ImageDTO | null) => ({ payload: { layerId: INITIAL_IMAGE_LAYER_ID, imageDTO } }),
    },
    iiLayerRecalled: (state, action: PayloadAction<InitialImageLayer>) => {
      state.layers = state.layers.filter((l) => (isInitialImageLayer(l) ? false : true));
      state.layers.push({ ...action.payload, isSelected: true });
      exclusivelySelectLayer(state, action.payload.id);
    },
    iiLayerImageChanged: (state, action: PayloadAction<{ layerId: string; imageDTO: ImageDTO | null }>) => {
      const { layerId, imageDTO } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isInitialImageLayer);
      layer.bbox = null;
      layer.bboxNeedsUpdate = true;
      layer.isEnabled = true;
      layer.image = imageDTO ? imageDTOToImageWithDims(imageDTO) : null;
    },
    iiLayerDenoisingStrengthChanged: (state, action: PayloadAction<{ layerId: string; denoisingStrength: number }>) => {
      const { layerId, denoisingStrength } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isInitialImageLayer);
      layer.denoisingStrength = denoisingStrength;
    },
    //#endregion

    //#region Raster Layers
    rasterLayerAdded: {
      reducer: (state, action: PayloadAction<{ layerId: string }>) => {
        const { layerId } = action.payload;
        const layer: RasterLayer = {
          id: getRasterLayerId(layerId),
          type: 'raster_layer',
          isEnabled: true,
          bbox: null,
          bboxNeedsUpdate: false,
          objects: [],
          opacity: 1,
          x: 0,
          y: 0,
          isSelected: true,
        };
        state.layers.push(layer);
        exclusivelySelectLayer(state, layer.id);
      },
      prepare: () => ({ payload: { layerId: uuidv4() } }),
    },
    //#endregion

    //#region Objects
    brushLineAdded: {
      reducer: (
        state,
        action: PayloadAction<
          AddBrushLineArg & {
            lineUuid: string;
          }
        >
      ) => {
        const { layerId, points, lineUuid, color } = action.payload;
        const layer = selectLayerOrThrow(state, layerId, isRGOrRasterlayer);
        layer.objects.push({
          id: getBrushLineId(layer.id, lineUuid),
          type: 'brush_line',
          // Points must be offset by the layer's x and y coordinates
          // TODO: Handle this in the event listener?
          points: [points[0] - layer.x, points[1] - layer.y, points[2] - layer.x, points[3] - layer.y],
          strokeWidth: state.brushSize,
          color,
        });
        layer.bboxNeedsUpdate = true;
        if (layer.type === 'regional_guidance_layer') {
          layer.uploadedMaskImage = null;
        }
      },
      prepare: (payload: AddBrushLineArg) => ({
        payload: { ...payload, lineUuid: uuidv4() },
      }),
    },
    eraserLineAdded: {
      reducer: (
        state,
        action: PayloadAction<
          AddEraserLineArg & {
            lineUuid: string;
          }
        >
      ) => {
        const { layerId, points, lineUuid } = action.payload;
        const layer = selectLayerOrThrow(state, layerId, isRGOrRasterlayer);
        layer.objects.push({
          id: getEraserLineId(layer.id, lineUuid),
          type: 'eraser_line',
          // Points must be offset by the layer's x and y coordinates
          // TODO: Handle this in the event listener?
          points: [points[0] - layer.x, points[1] - layer.y, points[2] - layer.x, points[3] - layer.y],
          strokeWidth: state.brushSize,
        });
        layer.bboxNeedsUpdate = true;
        if (isRegionalGuidanceLayer(layer)) {
          layer.uploadedMaskImage = null;
        }
      },
      prepare: (payload: AddEraserLineArg) => ({
        payload: { ...payload, lineUuid: uuidv4() },
      }),
    },
    linePointsAdded: (state, action: PayloadAction<AddPointToLineArg>) => {
      const { layerId, point } = action.payload;
      const layer = selectLayerOrThrow(state, layerId, isRGOrRasterlayer);
      const lastLine = layer.objects.findLast(isLine);
      if (!lastLine || !isLine(lastLine)) {
        return;
      }
      // Points must be offset by the layer's x and y coordinates
      // TODO: Handle this in the event listener
      lastLine.points.push(point[0] - layer.x, point[1] - layer.y);
      layer.bboxNeedsUpdate = true;
      if (isRegionalGuidanceLayer(layer)) {
        layer.uploadedMaskImage = null;
      }
    },
    rectAdded: {
      reducer: (state, action: PayloadAction<AddRectShapeArg & { rectUuid: string }>) => {
        const { layerId, rect, rectUuid, color } = action.payload;
        if (rect.height === 0 || rect.width === 0) {
          // Ignore zero-area rectangles
          return;
        }
        const layer = selectLayerOrThrow(state, layerId, isRGOrRasterlayer);
        const id = getRectId(layer.id, rectUuid);
        layer.objects.push({
          type: 'rect_shape',
          id,
          x: rect.x - layer.x,
          y: rect.y - layer.y,
          width: rect.width,
          height: rect.height,
          color,
        });
        layer.bboxNeedsUpdate = true;
        if (isRegionalGuidanceLayer(layer)) {
          layer.uploadedMaskImage = null;
        }
      },
      prepare: (payload: AddRectShapeArg) => ({ payload: { ...payload, rectUuid: uuidv4() } }),
    },
    //#endregion

    //#region Globals
    positivePromptChanged: (state, action: PayloadAction<string>) => {
      state.positivePrompt = action.payload;
    },
    negativePromptChanged: (state, action: PayloadAction<string>) => {
      state.negativePrompt = action.payload;
    },
    positivePrompt2Changed: (state, action: PayloadAction<string>) => {
      state.positivePrompt2 = action.payload;
    },
    negativePrompt2Changed: (state, action: PayloadAction<string>) => {
      state.negativePrompt2 = action.payload;
    },
    shouldConcatPromptsChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldConcatPrompts = action.payload;
    },
    widthChanged: (state, action: PayloadAction<{ width: number; updateAspectRatio?: boolean; clamp?: boolean }>) => {
      const { width, updateAspectRatio, clamp } = action.payload;
      state.size.width = clamp ? Math.max(roundDownToMultiple(width, 8), 64) : width;
      if (updateAspectRatio) {
        state.size.aspectRatio.value = state.size.width / state.size.height;
        state.size.aspectRatio.id = 'Free';
        state.size.aspectRatio.isLocked = false;
      }
    },
    heightChanged: (state, action: PayloadAction<{ height: number; updateAspectRatio?: boolean; clamp?: boolean }>) => {
      const { height, updateAspectRatio, clamp } = action.payload;
      state.size.height = clamp ? Math.max(roundDownToMultiple(height, 8), 64) : height;
      if (updateAspectRatio) {
        state.size.aspectRatio.value = state.size.width / state.size.height;
        state.size.aspectRatio.id = 'Free';
        state.size.aspectRatio.isLocked = false;
      }
    },
    aspectRatioChanged: (state, action: PayloadAction<AspectRatioState>) => {
      state.size.aspectRatio = action.payload;
    },
    brushSizeChanged: (state, action: PayloadAction<number>) => {
      state.brushSize = Math.round(action.payload);
    },
    brushColorChanged: (state, action: PayloadAction<RgbaColor>) => {
      state.brushColor = action.payload;
    },
    globalMaskLayerOpacityChanged: (state, action: PayloadAction<number>) => {
      state.globalMaskLayerOpacity = action.payload;
    },
    undo: (state) => {
      // Invalidate the bbox for all layers to prevent stale bboxes
      for (const layer of state.layers.filter(isRenderableLayer)) {
        layer.bboxNeedsUpdate = true;
      }
    },
    redo: (state) => {
      // Invalidate the bbox for all layers to prevent stale bboxes
      for (const layer of state.layers.filter(isRenderableLayer)) {
        layer.bboxNeedsUpdate = true;
      }
    },
    //#endregion
  },
  extraReducers(builder) {
    builder.addCase(modelChanged, (state, action) => {
      const newModel = action.payload;
      if (!newModel || action.meta.previousModel?.base === newModel.base) {
        // Model was cleared or the base didn't change
        return;
      }
      const optimalDimension = getOptimalDimension(newModel);
      if (getIsSizeOptimal(state.size.width, state.size.height, optimalDimension)) {
        return;
      }
      const { width, height } = calculateNewSize(state.size.aspectRatio.value, optimalDimension * optimalDimension);
      state.size.width = width;
      state.size.height = height;
    });

    // // TODO: This is a temp fix to reduce issues with T2I adapter having a different downscaling
    // // factor than the UNet. Hopefully we get an upstream fix in diffusers.
    // builder.addMatcher(isAnyControlAdapterAdded, (state, action) => {
    //   if (action.payload.type === 't2i_adapter') {
    //     state.size.width = roundToMultiple(state.size.width, 64);
    //     state.size.height = roundToMultiple(state.size.height, 64);
    //   }
    // });
  },
});

/**
 * This class is used to cycle through a set of colors for the prompt region layers.
 */
class LayerColors {
  static COLORS: RgbColor[] = [
    { r: 121, g: 157, b: 219 }, // rgb(121, 157, 219)
    { r: 131, g: 214, b: 131 }, // rgb(131, 214, 131)
    { r: 250, g: 225, b: 80 }, // rgb(250, 225, 80)
    { r: 220, g: 144, b: 101 }, // rgb(220, 144, 101)
    { r: 224, g: 117, b: 117 }, // rgb(224, 117, 117)
    { r: 213, g: 139, b: 202 }, // rgb(213, 139, 202)
    { r: 161, g: 120, b: 214 }, // rgb(161, 120, 214)
  ];
  static i = this.COLORS.length - 1;
  /**
   * Get the next color in the sequence. If a known color is provided, the next color will be the one after it.
   */
  static next(currentColor?: RgbColor): RgbColor {
    if (currentColor) {
      const i = this.COLORS.findIndex((c) => isEqual(c, currentColor));
      if (i !== -1) {
        this.i = i;
      }
    }
    this.i = (this.i + 1) % this.COLORS.length;
    const color = this.COLORS[this.i];
    assert(color);
    return color;
  }
}

export const {
  // Any Layer Type
  layerSelected,
  layerIsEnabledToggled,
  layerTranslated,
  layerBboxChanged,
  layerReset,
  layerDeleted,
  layerOpacityChanged,
  layerMovedForward,
  layerMovedToFront,
  layerMovedBackward,
  layerMovedToBack,
  selectedLayerDeleted,
  allLayersDeleted,
  // CA Layers
  caLayerAdded,
  caLayerRecalled,
  caLayerImageChanged,
  caLayerProcessedImageChanged,
  caLayerModelChanged,
  caLayerControlModeChanged,
  caLayerProcessorConfigChanged,
  caLayerIsFilterEnabledChanged,
  caLayerProcessorPendingBatchIdChanged,
  // IPA Layers
  ipaLayerAdded,
  ipaLayerRecalled,
  ipaLayerImageChanged,
  ipaLayerMethodChanged,
  ipaLayerModelChanged,
  ipaLayerCLIPVisionModelChanged,
  // CA or IPA Layers
  caOrIPALayerWeightChanged,
  caOrIPALayerBeginEndStepPctChanged,
  // RG Layers
  rgLayerAdded,
  rgLayerRecalled,
  rgLayerPositivePromptChanged,
  rgLayerNegativePromptChanged,
  rgLayerPreviewColorChanged,
  brushLineAdded,
  eraserLineAdded,
  linePointsAdded,
  rectAdded,
  rgLayerMaskImageUploaded,
  rgLayerAutoNegativeChanged,
  rgLayerIPAdapterAdded,
  rgLayerIPAdapterDeleted,
  rgLayerIPAdapterImageChanged,
  rgLayerIPAdapterWeightChanged,
  rgLayerIPAdapterBeginEndStepPctChanged,
  rgLayerIPAdapterMethodChanged,
  rgLayerIPAdapterModelChanged,
  rgLayerIPAdapterCLIPVisionModelChanged,
  // II Layer
  iiLayerAdded,
  iiLayerRecalled,
  iiLayerImageChanged,
  iiLayerDenoisingStrengthChanged,
  // Raster layers
  rasterLayerAdded,
  // Globals
  positivePromptChanged,
  negativePromptChanged,
  positivePrompt2Changed,
  negativePrompt2Changed,
  shouldConcatPromptsChanged,
  widthChanged,
  heightChanged,
  aspectRatioChanged,
  brushSizeChanged,
  brushColorChanged,
  globalMaskLayerOpacityChanged,
  undo,
  redo,
} = controlLayersSlice.actions;

export const selectControlLayersSlice = (state: RootState) => state.controlLayers;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrateControlLayersState = (state: any): any => {
  if (state._version === 1) {
    // Reset state for users on v1 (e.g. beta users), some changes could cause
    state = deepClone(initialControlLayersState);
  }
  if (state._version === 2) {
    // The CA `isProcessingImage` flag was replaced with a `processorPendingBatchId` property, fix up CA layers
    for (const layer of (state as ControlLayersState).layers) {
      if (layer.type === 'control_adapter_layer') {
        layer.controlAdapter.processorPendingBatchId = null;
        unset(layer.controlAdapter, 'isProcessingImage');
      }
    }
  }
  return state;
};

// Ephemeral interaction state
export const $isDrawing = atom(false);
export const $lastMouseDownPos = atom<Vector2d | null>(null);
export const $tool = atom<Tool>('brush');
export const $lastCursorPos = atom<Vector2d | null>(null);
export const $isPreviewVisible = atom(true);
export const $lastAddedPoint = atom<Vector2d | null>(null);

// Some nanostores that are manually synced to redux state to provide imperative access
// TODO(psyche): This is a hack, figure out another way to handle this...
export const $brushSize = atom<number>(0);
export const $brushColor = atom<RgbaColor>(DEFAULT_RGBA_COLOR);
export const $brushSpacingPx = atom<number>(0);
export const $selectedLayer = atom<Layer | null>(null);
export const $shouldInvertBrushSizeScrollDirection = atom(false);

export const controlLayersPersistConfig: PersistConfig<ControlLayersState> = {
  name: controlLayersSlice.name,
  initialState: initialControlLayersState,
  migrate: migrateControlLayersState,
  persistDenylist: [],
};

// These actions are _individually_ grouped together as single undoable actions
const undoableGroupByMatcher = isAnyOf(
  layerTranslated,
  brushSizeChanged,
  globalMaskLayerOpacityChanged,
  positivePromptChanged,
  negativePromptChanged,
  positivePrompt2Changed,
  negativePrompt2Changed,
  rgLayerPositivePromptChanged,
  rgLayerNegativePromptChanged,
  rgLayerPreviewColorChanged
);

// These are used to group actions into logical lines below (hate typos)
const LINE_1 = 'LINE_1';
const LINE_2 = 'LINE_2';

export const controlLayersUndoableConfig: UndoableOptions<ControlLayersState, UnknownAction> = {
  limit: 64,
  undoType: controlLayersSlice.actions.undo.type,
  redoType: controlLayersSlice.actions.redo.type,
  groupBy: (action, state, history) => {
    // Lines are started with `rgLayerLineAdded` and may have any number of subsequent `rgLayerPointsAdded` events.
    // We can use a double-buffer-esque trick to group each "logical" line as a single undoable action, without grouping
    // separate logical lines as a single undo action.
    if (brushLineAdded.match(action)) {
      return history.group === LINE_1 ? LINE_2 : LINE_1;
    }
    if (linePointsAdded.match(action)) {
      if (history.group === LINE_1 || history.group === LINE_2) {
        return history.group;
      }
    }
    if (undoableGroupByMatcher(action)) {
      return action.type;
    }
    return null;
  },
  filter: (action, _state, _history) => {
    // TODO(psyche): TEMP OVERRIDE
    return false;
    // // Ignore all actions from other slices
    // if (!action.type.startsWith(controlLayersSlice.name)) {
    //   return false;
    // }
    // // This action is triggered on state changes, including when we undo. If we do not ignore this action, when we
    // // undo, this action triggers and empties the future states array. Therefore, we must ignore this action.
    // if (layerBboxChanged.match(action)) {
    //   return false;
    // }
    // return true;
  },
};
