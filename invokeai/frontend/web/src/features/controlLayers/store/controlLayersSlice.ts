import type { PayloadAction, UnknownAction } from '@reduxjs/toolkit';
import { createSlice, isAnyOf } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { moveBackward, moveForward, moveToBack, moveToFront } from 'common/util/arrayUtils';
import { deepClone } from 'common/util/deepClone';
import type {
  CLIPVisionModel,
  ControlMode,
  ControlNetConfig,
  IPAdapterConfig,
  IPMethod,
  ProcessorConfig,
  T2IAdapterConfig,
} from 'features/controlLayers/util/controlAdapters';
import { buildControlAdapterProcessor, imageDTOToImageWithDims } from 'features/controlLayers/util/controlAdapters';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { calculateNewSize } from 'features/parameters/components/ImageSize/calculateNewSize';
import { initialAspectRatioState } from 'features/parameters/components/ImageSize/constants';
import type { AspectRatioState } from 'features/parameters/components/ImageSize/types';
import { modelChanged } from 'features/parameters/store/generationSlice';
import type { ParameterAutoNegative } from 'features/parameters/types/parameterSchemas';
import { getIsSizeOptimal, getOptimalDimension } from 'features/parameters/util/optimalDimension';
import type { IRect, Vector2d } from 'konva/lib/types';
import { isEqual, partition } from 'lodash-es';
import { atom } from 'nanostores';
import type { RgbColor } from 'react-colorful';
import type { UndoableOptions } from 'redux-undo';
import type { ControlNetModelConfig, ImageDTO, IPAdapterModelConfig, T2IAdapterModelConfig } from 'services/api/types';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

import type {
  ControlAdapterLayer,
  ControlLayersState,
  DrawingTool,
  IPAdapterLayer,
  Layer,
  RegionalGuidanceLayer,
  Tool,
  VectorMaskLine,
  VectorMaskRect,
} from './types';

export const initialControlLayersState: ControlLayersState = {
  _version: 1,
  selectedLayerId: null,
  brushSize: 100,
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

const isLine = (obj: VectorMaskLine | VectorMaskRect): obj is VectorMaskLine => obj.type === 'vector_mask_line';
export const isRegionalGuidanceLayer = (layer?: Layer): layer is RegionalGuidanceLayer =>
  layer?.type === 'regional_guidance_layer';
export const isControlAdapterLayer = (layer?: Layer): layer is ControlAdapterLayer =>
  layer?.type === 'control_adapter_layer';
export const isIPAdapterLayer = (layer?: Layer): layer is IPAdapterLayer => layer?.type === 'ip_adapter_layer';
export const isRenderableLayer = (layer?: Layer): layer is RegionalGuidanceLayer | ControlAdapterLayer =>
  layer?.type === 'regional_guidance_layer' || layer?.type === 'control_adapter_layer';
const resetLayer = (layer: Layer) => {
  if (layer.type === 'regional_guidance_layer') {
    layer.maskObjects = [];
    layer.bbox = null;
    layer.isEnabled = true;
    layer.needsPixelBbox = false;
    layer.bboxNeedsUpdate = false;
    return;
  }
};

export const selectCALayerOrThrow = (state: ControlLayersState, layerId: string): ControlAdapterLayer => {
  const layer = state.layers.find((l) => l.id === layerId);
  assert(isControlAdapterLayer(layer));
  return layer;
};
export const selectIPALayerOrThrow = (state: ControlLayersState, layerId: string): IPAdapterLayer => {
  const layer = state.layers.find((l) => l.id === layerId);
  assert(isIPAdapterLayer(layer));
  return layer;
};
export const selectCAOrIPALayerOrThrow = (
  state: ControlLayersState,
  layerId: string
): ControlAdapterLayer | IPAdapterLayer => {
  const layer = state.layers.find((l) => l.id === layerId);
  assert(isControlAdapterLayer(layer) || isIPAdapterLayer(layer));
  return layer;
};
export const selectRGLayerOrThrow = (state: ControlLayersState, layerId: string): RegionalGuidanceLayer => {
  const layer = state.layers.find((l) => l.id === layerId);
  assert(isRegionalGuidanceLayer(layer));
  return layer;
};
export const selectRGLayerIPAdapterOrThrow = (
  state: ControlLayersState,
  layerId: string,
  ipAdapterId: string
): IPAdapterConfig => {
  const layer = state.layers.find((l) => l.id === layerId);
  assert(isRegionalGuidanceLayer(layer));
  const ipAdapter = layer.ipAdapters.find((ipAdapter) => ipAdapter.id === ipAdapterId);
  assert(ipAdapter);
  return ipAdapter;
};
const getVectorMaskPreviewColor = (state: ControlLayersState): RgbColor => {
  const rgLayers = state.layers.filter(isRegionalGuidanceLayer);
  const lastColor = rgLayers[rgLayers.length - 1]?.previewColor;
  return LayerColors.next(lastColor);
};

export const controlLayersSlice = createSlice({
  name: 'controlLayers',
  initialState: initialControlLayersState,
  reducers: {
    //#region Any Layer Type
    layerSelected: (state, action: PayloadAction<string>) => {
      for (const layer of state.layers.filter(isRenderableLayer)) {
        if (layer.id === action.payload) {
          layer.isSelected = true;
          state.selectedLayerId = action.payload;
        } else {
          layer.isSelected = false;
        }
      }
    },
    layerVisibilityToggled: (state, action: PayloadAction<string>) => {
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
    },
    layerBboxChanged: (state, action: PayloadAction<{ layerId: string; bbox: IRect | null }>) => {
      const { layerId, bbox } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (isRenderableLayer(layer)) {
        layer.bbox = bbox;
        layer.bboxNeedsUpdate = false;
        if (bbox === null && layer.type === 'regional_guidance_layer') {
          // The layer was fully erased, empty its objects to prevent accumulation of invisible objects
          layer.maskObjects = [];
          layer.needsPixelBbox = false;
        }
      }
    },
    layerReset: (state, action: PayloadAction<string>) => {
      const layer = state.layers.find((l) => l.id === action.payload);
      if (layer) {
        resetLayer(layer);
      }
    },
    layerDeleted: (state, action: PayloadAction<string>) => {
      state.layers = state.layers.filter((l) => l.id !== action.payload);
      state.selectedLayerId = state.layers[0]?.id ?? null;
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
    selectedLayerReset: (state) => {
      const layer = state.layers.find((l) => l.id === state.selectedLayerId);
      if (layer) {
        resetLayer(layer);
      }
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
        action: PayloadAction<{ layerId: string; controlAdapter: ControlNetConfig | T2IAdapterConfig }>
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
        state.selectedLayerId = layer.id;
        for (const layer of state.layers.filter(isRenderableLayer)) {
          if (layer.id !== layerId) {
            layer.isSelected = false;
          }
        }
      },
      prepare: (controlAdapter: ControlNetConfig | T2IAdapterConfig) => ({
        payload: { layerId: uuidv4(), controlAdapter },
      }),
    },
    caLayerImageChanged: (state, action: PayloadAction<{ layerId: string; imageDTO: ImageDTO | null }>) => {
      const { layerId, imageDTO } = action.payload;
      const layer = selectCALayerOrThrow(state, layerId);
      layer.bbox = null;
      layer.bboxNeedsUpdate = true;
      layer.isEnabled = true;
      layer.controlAdapter.image = imageDTO ? imageDTOToImageWithDims(imageDTO) : null;
      layer.controlAdapter.processedImage = null;
    },
    caLayerProcessedImageChanged: (state, action: PayloadAction<{ layerId: string; imageDTO: ImageDTO | null }>) => {
      const { layerId, imageDTO } = action.payload;
      const layer = selectCALayerOrThrow(state, layerId);
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
      const layer = selectCALayerOrThrow(state, layerId);
      if (!modelConfig) {
        layer.controlAdapter.model = null;
        return;
      }
      layer.controlAdapter.model = zModelIdentifierField.parse(modelConfig);
      const candidateProcessorConfig = buildControlAdapterProcessor(modelConfig);
      if (candidateProcessorConfig?.type !== layer.controlAdapter.processorConfig?.type) {
        // The processor has changed. For example, the previous model was a Canny model and the new model is a Depth
        // model. We need to use the new processor.
        layer.controlAdapter.processedImage = null;
        layer.controlAdapter.processorConfig = candidateProcessorConfig;
      }
    },
    caLayerControlModeChanged: (state, action: PayloadAction<{ layerId: string; controlMode: ControlMode }>) => {
      const { layerId, controlMode } = action.payload;
      const layer = selectCALayerOrThrow(state, layerId);
      assert(layer.controlAdapter.type === 'controlnet');
      layer.controlAdapter.controlMode = controlMode;
    },
    caLayerProcessorConfigChanged: (
      state,
      action: PayloadAction<{ layerId: string; processorConfig: ProcessorConfig | null }>
    ) => {
      const { layerId, processorConfig } = action.payload;
      const layer = selectCALayerOrThrow(state, layerId);
      layer.controlAdapter.processorConfig = processorConfig;
    },
    caLayerIsFilterEnabledChanged: (state, action: PayloadAction<{ layerId: string; isFilterEnabled: boolean }>) => {
      const { layerId, isFilterEnabled } = action.payload;
      const layer = selectCALayerOrThrow(state, layerId);
      layer.isFilterEnabled = isFilterEnabled;
    },
    caLayerOpacityChanged: (state, action: PayloadAction<{ layerId: string; opacity: number }>) => {
      const { layerId, opacity } = action.payload;
      const layer = selectCALayerOrThrow(state, layerId);
      layer.opacity = opacity;
    },
    caLayerIsProcessingImageChanged: (
      state,
      action: PayloadAction<{ layerId: string; isProcessingImage: boolean }>
    ) => {
      const { layerId, isProcessingImage } = action.payload;
      const layer = selectCALayerOrThrow(state, layerId);
      layer.controlAdapter.isProcessingImage = isProcessingImage;
    },
    //#endregion

    //#region IP Adapter Layers
    ipaLayerAdded: {
      reducer: (state, action: PayloadAction<{ layerId: string; ipAdapter: IPAdapterConfig }>) => {
        const { layerId, ipAdapter } = action.payload;
        const layer: IPAdapterLayer = {
          id: getIPALayerId(layerId),
          type: 'ip_adapter_layer',
          isEnabled: true,
          ipAdapter,
        };
        state.layers.push(layer);
      },
      prepare: (ipAdapter: IPAdapterConfig) => ({ payload: { layerId: uuidv4(), ipAdapter } }),
    },
    ipaLayerImageChanged: (state, action: PayloadAction<{ layerId: string; imageDTO: ImageDTO | null }>) => {
      const { layerId, imageDTO } = action.payload;
      const layer = selectIPALayerOrThrow(state, layerId);
      layer.ipAdapter.image = imageDTO ? imageDTOToImageWithDims(imageDTO) : null;
    },
    ipaLayerWeightChanged: (state, action: PayloadAction<{ layerId: string; weight: number }>) => {
      const { layerId, weight } = action.payload;
      const layer = selectIPALayerOrThrow(state, layerId);
      layer.ipAdapter.weight = weight;
    },
    ipaLayerBeginEndStepPctChanged: (
      state,
      action: PayloadAction<{ layerId: string; beginEndStepPct: [number, number] }>
    ) => {
      const { layerId, beginEndStepPct } = action.payload;
      const layer = selectIPALayerOrThrow(state, layerId);
      layer.ipAdapter.beginEndStepPct = beginEndStepPct;
    },
    ipaLayerMethodChanged: (state, action: PayloadAction<{ layerId: string; method: IPMethod }>) => {
      const { layerId, method } = action.payload;
      const layer = selectIPALayerOrThrow(state, layerId);
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
      const layer = selectIPALayerOrThrow(state, layerId);
      if (!modelConfig) {
        layer.ipAdapter.model = null;
        return;
      }
      layer.ipAdapter.model = zModelIdentifierField.parse(modelConfig);
    },
    ipaLayerCLIPVisionModelChanged: (
      state,
      action: PayloadAction<{ layerId: string; clipVisionModel: CLIPVisionModel }>
    ) => {
      const { layerId, clipVisionModel } = action.payload;
      const layer = selectIPALayerOrThrow(state, layerId);
      layer.ipAdapter.clipVisionModel = clipVisionModel;
    },
    //#endregion

    //#region CA or IPA Layers
    caOrIPALayerWeightChanged: (state, action: PayloadAction<{ layerId: string; weight: number }>) => {
      const { layerId, weight } = action.payload;
      const layer = selectCAOrIPALayerOrThrow(state, layerId);
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
      const layer = selectCAOrIPALayerOrThrow(state, layerId);
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
          maskObjects: [],
          previewColor: getVectorMaskPreviewColor(state),
          x: 0,
          y: 0,
          autoNegative: 'invert',
          needsPixelBbox: false,
          positivePrompt: '',
          negativePrompt: null,
          ipAdapters: [],
          isSelected: true,
        };
        state.layers.push(layer);
        state.selectedLayerId = layer.id;
        for (const layer of state.layers.filter(isRenderableLayer)) {
          if (layer.id !== layerId) {
            layer.isSelected = false;
          }
        }
      },
      prepare: () => ({ payload: { layerId: uuidv4() } }),
    },
    rgLayerPositivePromptChanged: (state, action: PayloadAction<{ layerId: string; prompt: string | null }>) => {
      const { layerId, prompt } = action.payload;
      const layer = selectRGLayerOrThrow(state, layerId);
      layer.positivePrompt = prompt;
    },
    rgLayerNegativePromptChanged: (state, action: PayloadAction<{ layerId: string; prompt: string | null }>) => {
      const { layerId, prompt } = action.payload;
      const layer = selectRGLayerOrThrow(state, layerId);
      layer.negativePrompt = prompt;
    },
    rgLayerPreviewColorChanged: (state, action: PayloadAction<{ layerId: string; color: RgbColor }>) => {
      const { layerId, color } = action.payload;
      const layer = selectRGLayerOrThrow(state, layerId);
      layer.previewColor = color;
    },
    rgLayerLineAdded: {
      reducer: (
        state,
        action: PayloadAction<{
          layerId: string;
          points: [number, number, number, number];
          tool: DrawingTool;
          lineUuid: string;
        }>
      ) => {
        const { layerId, points, tool, lineUuid } = action.payload;
        const layer = selectRGLayerOrThrow(state, layerId);
        const lineId = getRGLayerLineId(layer.id, lineUuid);
        layer.maskObjects.push({
          type: 'vector_mask_line',
          tool: tool,
          id: lineId,
          // Points must be offset by the layer's x and y coordinates
          // TODO: Handle this in the event listener?
          points: [points[0] - layer.x, points[1] - layer.y, points[2] - layer.x, points[3] - layer.y],
          strokeWidth: state.brushSize,
        });
        layer.bboxNeedsUpdate = true;
        if (!layer.needsPixelBbox && tool === 'eraser') {
          layer.needsPixelBbox = true;
        }
      },
      prepare: (payload: { layerId: string; points: [number, number, number, number]; tool: DrawingTool }) => ({
        payload: { ...payload, lineUuid: uuidv4() },
      }),
    },
    rgLayerPointsAdded: (state, action: PayloadAction<{ layerId: string; point: [number, number] }>) => {
      const { layerId, point } = action.payload;
      const layer = selectRGLayerOrThrow(state, layerId);
      const lastLine = layer.maskObjects.findLast(isLine);
      if (!lastLine) {
        return;
      }
      // Points must be offset by the layer's x and y coordinates
      // TODO: Handle this in the event listener
      lastLine.points.push(point[0] - layer.x, point[1] - layer.y);
      layer.bboxNeedsUpdate = true;
    },
    rgLayerRectAdded: {
      reducer: (state, action: PayloadAction<{ layerId: string; rect: IRect; rectUuid: string }>) => {
        const { layerId, rect, rectUuid } = action.payload;
        if (rect.height === 0 || rect.width === 0) {
          // Ignore zero-area rectangles
          return;
        }
        const layer = selectRGLayerOrThrow(state, layerId);
        const id = getRGLayerRectId(layer.id, rectUuid);
        layer.maskObjects.push({
          type: 'vector_mask_rect',
          id,
          x: rect.x - layer.x,
          y: rect.y - layer.y,
          width: rect.width,
          height: rect.height,
        });
        layer.bboxNeedsUpdate = true;
      },
      prepare: (payload: { layerId: string; rect: IRect }) => ({ payload: { ...payload, rectUuid: uuidv4() } }),
    },
    rgLayerAutoNegativeChanged: (
      state,
      action: PayloadAction<{ layerId: string; autoNegative: ParameterAutoNegative }>
    ) => {
      const { layerId, autoNegative } = action.payload;
      const layer = selectRGLayerOrThrow(state, layerId);
      layer.autoNegative = autoNegative;
    },
    rgLayerIPAdapterAdded: (state, action: PayloadAction<{ layerId: string; ipAdapter: IPAdapterConfig }>) => {
      const { layerId, ipAdapter } = action.payload;
      const layer = selectRGLayerOrThrow(state, layerId);
      layer.ipAdapters.push(ipAdapter);
    },
    rgLayerIPAdapterDeleted: (state, action: PayloadAction<{ layerId: string; ipAdapterId: string }>) => {
      const { layerId, ipAdapterId } = action.payload;
      const layer = selectRGLayerOrThrow(state, layerId);
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
      action: PayloadAction<{ layerId: string; ipAdapterId: string; method: IPMethod }>
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
      action: PayloadAction<{ layerId: string; ipAdapterId: string; clipVisionModel: CLIPVisionModel }>
    ) => {
      const { layerId, ipAdapterId, clipVisionModel } = action.payload;
      const ipAdapter = selectRGLayerIPAdapterOrThrow(state, layerId, ipAdapterId);
      ipAdapter.clipVisionModel = clipVisionModel;
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
    widthChanged: (state, action: PayloadAction<{ width: number; updateAspectRatio?: boolean }>) => {
      const { width, updateAspectRatio } = action.payload;
      state.size.width = width;
      if (updateAspectRatio) {
        state.size.aspectRatio.value = width / state.size.height;
        state.size.aspectRatio.id = 'Free';
        state.size.aspectRatio.isLocked = false;
      }
    },
    heightChanged: (state, action: PayloadAction<{ height: number; updateAspectRatio?: boolean }>) => {
      const { height, updateAspectRatio } = action.payload;
      state.size.height = height;
      if (updateAspectRatio) {
        state.size.aspectRatio.value = state.size.width / height;
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
  layerVisibilityToggled,
  layerTranslated,
  layerBboxChanged,
  layerReset,
  layerDeleted,
  layerMovedForward,
  layerMovedToFront,
  layerMovedBackward,
  layerMovedToBack,
  selectedLayerReset,
  selectedLayerDeleted,
  allLayersDeleted,
  // CA Layers
  caLayerAdded,
  caLayerImageChanged,
  caLayerProcessedImageChanged,
  caLayerModelChanged,
  caLayerControlModeChanged,
  caLayerProcessorConfigChanged,
  caLayerIsFilterEnabledChanged,
  caLayerOpacityChanged,
  caLayerIsProcessingImageChanged,
  // IPA Layers
  ipaLayerAdded,
  ipaLayerImageChanged,
  ipaLayerWeightChanged,
  ipaLayerBeginEndStepPctChanged,
  ipaLayerMethodChanged,
  ipaLayerModelChanged,
  ipaLayerCLIPVisionModelChanged,
  // CA or IPA Layers
  caOrIPALayerWeightChanged,
  caOrIPALayerBeginEndStepPctChanged,
  // RG Layers
  rgLayerAdded,
  rgLayerPositivePromptChanged,
  rgLayerNegativePromptChanged,
  rgLayerPreviewColorChanged,
  rgLayerLineAdded,
  rgLayerPointsAdded,
  rgLayerRectAdded,
  rgLayerAutoNegativeChanged,
  rgLayerIPAdapterAdded,
  rgLayerIPAdapterDeleted,
  rgLayerIPAdapterImageChanged,
  rgLayerIPAdapterWeightChanged,
  rgLayerIPAdapterBeginEndStepPctChanged,
  rgLayerIPAdapterMethodChanged,
  rgLayerIPAdapterModelChanged,
  rgLayerIPAdapterCLIPVisionModelChanged,
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
  globalMaskLayerOpacityChanged,
  undo,
  redo,
} = controlLayersSlice.actions;

export const selectControlLayersSlice = (state: RootState) => state.controlLayers;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrateControlLayersState = (state: any): any => {
  return state;
};

export const $isMouseDown = atom(false);
export const $isMouseOver = atom(false);
export const $lastMouseDownPos = atom<Vector2d | null>(null);
export const $tool = atom<Tool>('brush');
export const $cursorPosition = atom<Vector2d | null>(null);

// IDs for singleton Konva layers and objects
export const TOOL_PREVIEW_LAYER_ID = 'tool_preview_layer';
export const TOOL_PREVIEW_BRUSH_GROUP_ID = 'tool_preview_layer.brush_group';
export const TOOL_PREVIEW_BRUSH_FILL_ID = 'tool_preview_layer.brush_fill';
export const TOOL_PREVIEW_BRUSH_BORDER_INNER_ID = 'tool_preview_layer.brush_border_inner';
export const TOOL_PREVIEW_BRUSH_BORDER_OUTER_ID = 'tool_preview_layer.brush_border_outer';
export const TOOL_PREVIEW_RECT_ID = 'tool_preview_layer.rect';
export const BACKGROUND_LAYER_ID = 'background_layer';
export const BACKGROUND_RECT_ID = 'background_layer.rect';
export const NO_LAYERS_MESSAGE_LAYER_ID = 'no_layers_message';

// Names (aka classes) for Konva layers and objects
export const CA_LAYER_NAME = 'control_adapter_layer';
export const CA_LAYER_IMAGE_NAME = 'control_adapter_layer.image';
export const RG_LAYER_NAME = 'regional_guidance_layer';
export const RG_LAYER_LINE_NAME = 'regional_guidance_layer.line';
export const RG_LAYER_OBJECT_GROUP_NAME = 'regional_guidance_layer.object_group';
export const RG_LAYER_RECT_NAME = 'regional_guidance_layer.rect';
export const LAYER_BBOX_NAME = 'layer.bbox';

// Getters for non-singleton layer and object IDs
const getRGLayerId = (layerId: string) => `${RG_LAYER_NAME}_${layerId}`;
const getRGLayerLineId = (layerId: string, lineId: string) => `${layerId}.line_${lineId}`;
const getRGLayerRectId = (layerId: string, lineId: string) => `${layerId}.rect_${lineId}`;
export const getRGLayerObjectGroupId = (layerId: string, groupId: string) => `${layerId}.objectGroup_${groupId}`;
export const getLayerBboxId = (layerId: string) => `${layerId}.bbox`;
const getCALayerId = (layerId: string) => `control_adapter_layer_${layerId}`;
export const getCALayerImageId = (layerId: string, imageName: string) => `${layerId}.image_${imageName}`;
const getIPALayerId = (layerId: string) => `ip_adapter_layer_${layerId}`;

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
    if (rgLayerLineAdded.match(action)) {
      return history.group === LINE_1 ? LINE_2 : LINE_1;
    }
    if (rgLayerPointsAdded.match(action)) {
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
    // Ignore all actions from other slices
    if (!action.type.startsWith(controlLayersSlice.name)) {
      return false;
    }
    // This action is triggered on state changes, including when we undo. If we do not ignore this action, when we
    // undo, this action triggers and empties the future states array. Therefore, we must ignore this action.
    if (layerBboxChanged.match(action)) {
      return false;
    }
    return true;
  },
};
