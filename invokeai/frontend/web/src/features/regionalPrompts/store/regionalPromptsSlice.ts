import type { PayloadAction, UnknownAction } from '@reduxjs/toolkit';
import { createSlice, isAnyOf } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { moveBackward, moveForward, moveToBack, moveToFront } from 'common/util/arrayUtils';
import { controlAdapterRemoved } from 'features/controlAdapters/store/controlAdaptersSlice';
import type { ControlAdapterConfig } from 'features/controlAdapters/store/types';
import type {
  ParameterAutoNegative,
  ParameterNegativePrompt,
  ParameterNegativeStylePromptSDXL,
  ParameterPositivePrompt,
  ParameterPositiveStylePromptSDXL,
} from 'features/parameters/types/parameterSchemas';
import type { IRect, Vector2d } from 'konva/lib/types';
import { isEqual } from 'lodash-es';
import { atom } from 'nanostores';
import type { RgbColor } from 'react-colorful';
import type { UndoableOptions } from 'redux-undo';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

type DrawingTool = 'brush' | 'eraser';

export type Tool = DrawingTool | 'move' | 'rect';

export type VectorMaskLine = {
  id: string;
  type: 'vector_mask_line';
  tool: DrawingTool;
  strokeWidth: number;
  points: number[];
};

export type VectorMaskRect = {
  id: string;
  type: 'vector_mask_rect';
  x: number;
  y: number;
  width: number;
  height: number;
};

type LayerBase = {
  id: string;
  x: number;
  y: number;
  bbox: IRect | null;
  bboxNeedsUpdate: boolean;
  isVisible: boolean;
};

type ControlLayer = LayerBase & {
  type: 'control_layer';
  controlAdapter: ControlAdapterConfig;
};

type MaskLayerBase = LayerBase & {
  positivePrompt: string | null;
  negativePrompt: string | null; // Up to one text prompt per mask
  ipAdapterIds: string[]; // Any number of image prompts
  previewColor: RgbColor;
  autoNegative: ParameterAutoNegative;
  needsPixelBbox: boolean; // Needs the slower pixel-based bbox calculation - set to true when an there is an eraser object
};

export type VectorMaskLayer = MaskLayerBase & {
  type: 'vector_mask_layer';
  objects: (VectorMaskLine | VectorMaskRect)[];
};

export type Layer = VectorMaskLayer | ControlLayer;

type BaseLayerState = {
  positivePrompt: ParameterPositivePrompt;
  negativePrompt: ParameterNegativePrompt;
  positivePrompt2: ParameterPositiveStylePromptSDXL;
  negativePrompt2: ParameterNegativeStylePromptSDXL;
  shouldConcatPrompts: boolean;
};

type RegionalPromptsState = {
  _version: 1;
  selectedLayerId: string | null;
  layers: Layer[];
  brushSize: number;
  globalMaskLayerOpacity: number;
  isEnabled: boolean;
  baseLayer: BaseLayerState;
};

export const initialRegionalPromptsState: RegionalPromptsState = {
  _version: 1,
  selectedLayerId: null,
  brushSize: 100,
  layers: [],
  globalMaskLayerOpacity: 0.5, // this globally changes all mask layers' opacity
  isEnabled: true,
  baseLayer: {
    positivePrompt: '',
    negativePrompt: '',
    positivePrompt2: '',
    negativePrompt2: '',
    shouldConcatPrompts: true,
  },
};

const isLine = (obj: VectorMaskLine | VectorMaskRect): obj is VectorMaskLine => obj.type === 'vector_mask_line';
export const isVectorMaskLayer = (layer?: Layer): layer is VectorMaskLayer => layer?.type === 'vector_mask_layer';
const resetLayer = (layer: Layer) => {
  if (layer.type === 'vector_mask_layer') {
    layer.objects = [];
    layer.bbox = null;
    layer.isVisible = true;
    layer.needsPixelBbox = false;
    layer.bboxNeedsUpdate = false;
    return;
  }

  if (layer.type === 'control_layer') {
    // TODO
  }
};
const getVectorMaskPreviewColor = (state: RegionalPromptsState): RgbColor => {
  const vmLayers = state.layers.filter(isVectorMaskLayer);
  const lastColor = vmLayers[vmLayers.length - 1]?.previewColor;
  return LayerColors.next(lastColor);
};

export const regionalPromptsSlice = createSlice({
  name: 'regionalPrompts',
  initialState: initialRegionalPromptsState,
  reducers: {
    //#region All Layers
    layerAdded: {
      reducer: (state, action: PayloadAction<Layer['type'], string, { uuid: string }>) => {
        const type = action.payload;
        if (type === 'vector_mask_layer') {
          const layer: VectorMaskLayer = {
            id: getVectorMaskLayerId(action.meta.uuid),
            type,
            isVisible: true,
            bbox: null,
            bboxNeedsUpdate: false,
            objects: [],
            previewColor: getVectorMaskPreviewColor(state),
            x: 0,
            y: 0,
            autoNegative: 'invert',
            needsPixelBbox: false,
            positivePrompt: '',
            negativePrompt: null,
            ipAdapterIds: [],
          };
          state.layers.push(layer);
          state.selectedLayerId = layer.id;
          return;
        }

        if (type === 'control_layer') {
          // TODO
          return;
        }
      },
      prepare: (payload: Layer['type']) => ({ payload, meta: { uuid: uuidv4() } }),
    },
    layerSelected: (state, action: PayloadAction<string>) => {
      const layer = state.layers.find((l) => l.id === action.payload);
      if (layer) {
        state.selectedLayerId = layer.id;
      }
    },
    layerVisibilityToggled: (state, action: PayloadAction<string>) => {
      const layer = state.layers.find((l) => l.id === action.payload);
      if (layer) {
        layer.isVisible = !layer.isVisible;
      }
    },
    layerTranslated: (state, action: PayloadAction<{ layerId: string; x: number; y: number }>) => {
      const { layerId, x, y } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (layer) {
        layer.x = x;
        layer.y = y;
      }
    },
    layerBboxChanged: (state, action: PayloadAction<{ layerId: string; bbox: IRect | null }>) => {
      const { layerId, bbox } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (layer) {
        layer.bbox = bbox;
        layer.bboxNeedsUpdate = false;
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
      moveForward(state.layers, cb);
    },
    layerMovedToFront: (state, action: PayloadAction<string>) => {
      const cb = (l: Layer) => l.id === action.payload;
      // Because the layers are in reverse order, moving to the front is equivalent to moving to the back
      moveToBack(state.layers, cb);
    },
    layerMovedBackward: (state, action: PayloadAction<string>) => {
      const cb = (l: Layer) => l.id === action.payload;
      moveBackward(state.layers, cb);
    },
    layerMovedToBack: (state, action: PayloadAction<string>) => {
      const cb = (l: Layer) => l.id === action.payload;
      // Because the layers are in reverse order, moving to the back is equivalent to moving to the front
      moveToFront(state.layers, cb);
    },
    allLayersDeleted: (state) => {
      state.layers = [];
      state.selectedLayerId = null;
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
    //#endregion

    //#region Mask Layers
    maskLayerPositivePromptChanged: (state, action: PayloadAction<{ layerId: string; prompt: string | null }>) => {
      const { layerId, prompt } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (layer?.type === 'vector_mask_layer') {
        layer.positivePrompt = prompt;
      }
    },
    maskLayerNegativePromptChanged: (state, action: PayloadAction<{ layerId: string; prompt: string | null }>) => {
      const { layerId, prompt } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (layer?.type === 'vector_mask_layer') {
        layer.negativePrompt = prompt;
      }
    },
    maskLayerIPAdapterAdded: {
      reducer: (state, action: PayloadAction<string, string, { uuid: string }>) => {
        const layer = state.layers.find((l) => l.id === action.payload);
        if (layer?.type === 'vector_mask_layer') {
          layer.ipAdapterIds.push(action.meta.uuid);
        }
      },
      prepare: (payload: string) => ({ payload, meta: { uuid: uuidv4() } }),
    },
    maskLayerPreviewColorChanged: (state, action: PayloadAction<{ layerId: string; color: RgbColor }>) => {
      const { layerId, color } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (layer?.type === 'vector_mask_layer') {
        layer.previewColor = color;
      }
    },
    maskLayerLineAdded: {
      reducer: (
        state,
        action: PayloadAction<
          { layerId: string; points: [number, number, number, number]; tool: DrawingTool },
          string,
          { uuid: string }
        >
      ) => {
        const { layerId, points, tool } = action.payload;
        const layer = state.layers.find((l) => l.id === layerId);
        if (layer?.type === 'vector_mask_layer') {
          const lineId = getVectorMaskLayerLineId(layer.id, action.meta.uuid);
          layer.objects.push({
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
        }
      },
      prepare: (payload: { layerId: string; points: [number, number, number, number]; tool: DrawingTool }) => ({
        payload,
        meta: { uuid: uuidv4() },
      }),
    },
    maskLayerPointsAdded: (state, action: PayloadAction<{ layerId: string; point: [number, number] }>) => {
      const { layerId, point } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (layer?.type === 'vector_mask_layer') {
        const lastLine = layer.objects.findLast(isLine);
        if (!lastLine) {
          return;
        }
        // Points must be offset by the layer's x and y coordinates
        // TODO: Handle this in the event listener
        lastLine.points.push(point[0] - layer.x, point[1] - layer.y);
        layer.bboxNeedsUpdate = true;
      }
    },
    maskLayerRectAdded: {
      reducer: (state, action: PayloadAction<{ layerId: string; rect: IRect }, string, { uuid: string }>) => {
        const { layerId, rect } = action.payload;
        if (rect.height === 0 || rect.width === 0) {
          // Ignore zero-area rectangles
          return;
        }
        const layer = state.layers.find((l) => l.id === layerId);
        if (layer?.type === 'vector_mask_layer') {
          const id = getVectorMaskLayerRectId(layer.id, action.meta.uuid);
          layer.objects.push({
            type: 'vector_mask_rect',
            id,
            x: rect.x - layer.x,
            y: rect.y - layer.y,
            width: rect.width,
            height: rect.height,
          });
          layer.bboxNeedsUpdate = true;
        }
      },
      prepare: (payload: { layerId: string; rect: IRect }) => ({ payload, meta: { uuid: uuidv4() } }),
    },
    maskLayerAutoNegativeChanged: (
      state,
      action: PayloadAction<{ layerId: string; autoNegative: ParameterAutoNegative }>
    ) => {
      const { layerId, autoNegative } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (layer?.type === 'vector_mask_layer') {
        layer.autoNegative = autoNegative;
      }
    },
    //#endregion

    //#region Base Layer
    positivePromptChanged: (state, action: PayloadAction<string>) => {
      state.baseLayer.positivePrompt = action.payload;
    },
    negativePromptChanged: (state, action: PayloadAction<string>) => {
      state.baseLayer.negativePrompt = action.payload;
    },
    positivePrompt2Changed: (state, action: PayloadAction<string>) => {
      state.baseLayer.positivePrompt2 = action.payload;
    },
    negativePrompt2Changed: (state, action: PayloadAction<string>) => {
      state.baseLayer.negativePrompt2 = action.payload;
    },
    shouldConcatPromptsChanged: (state, action: PayloadAction<boolean>) => {
      state.baseLayer.shouldConcatPrompts = action.payload;
    },
    //#endregion

    //#region General
    brushSizeChanged: (state, action: PayloadAction<number>) => {
      state.brushSize = action.payload;
    },
    globalMaskLayerOpacityChanged: (state, action: PayloadAction<number>) => {
      state.globalMaskLayerOpacity = action.payload;
    },
    isEnabledChanged: (state, action: PayloadAction<boolean>) => {
      state.isEnabled = action.payload;
    },
    undo: (state) => {
      // Invalidate the bbox for all layers to prevent stale bboxes
      for (const layer of state.layers) {
        layer.bboxNeedsUpdate = true;
      }
    },
    redo: (state) => {
      // Invalidate the bbox for all layers to prevent stale bboxes
      for (const layer of state.layers) {
        layer.bboxNeedsUpdate = true;
      }
    },
    //#endregion
  },
  extraReducers(builder) {
    builder.addCase(controlAdapterRemoved, (state, action) => {
      state.layers.filter(isVectorMaskLayer).forEach((layer) => {
        layer.ipAdapterIds = layer.ipAdapterIds.filter((id) => id !== action.payload.id);
      });
    });
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
  // All layer actions
  layerAdded,
  layerDeleted,
  layerMovedBackward,
  layerMovedForward,
  layerMovedToBack,
  layerMovedToFront,
  layerReset,
  layerSelected,
  layerTranslated,
  layerBboxChanged,
  layerVisibilityToggled,
  allLayersDeleted,
  selectedLayerReset,
  selectedLayerDeleted,
  // Mask layer actions
  maskLayerLineAdded,
  maskLayerPointsAdded,
  maskLayerRectAdded,
  maskLayerNegativePromptChanged,
  maskLayerPositivePromptChanged,
  maskLayerIPAdapterAdded,
  maskLayerAutoNegativeChanged,
  maskLayerPreviewColorChanged,
  // Base layer actions
  positivePromptChanged,
  negativePromptChanged,
  positivePrompt2Changed,
  negativePrompt2Changed,
  shouldConcatPromptsChanged,
  // General actions
  brushSizeChanged,
  globalMaskLayerOpacityChanged,
  undo,
  redo,
} = regionalPromptsSlice.actions;

export const selectRegionalPromptsSlice = (state: RootState) => state.regionalPrompts;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrateRegionalPromptsState = (state: any): any => {
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

// Names (aka classes) for Konva layers and objects
export const VECTOR_MASK_LAYER_NAME = 'vector_mask_layer';
export const VECTOR_MASK_LAYER_LINE_NAME = 'vector_mask_layer.line';
export const VECTOR_MASK_LAYER_OBJECT_GROUP_NAME = 'vector_mask_layer.object_group';
export const VECTOR_MASK_LAYER_RECT_NAME = 'vector_mask_layer.rect';
export const LAYER_BBOX_NAME = 'layer.bbox';

// Getters for non-singleton layer and object IDs
const getVectorMaskLayerId = (layerId: string) => `${VECTOR_MASK_LAYER_NAME}_${layerId}`;
const getVectorMaskLayerLineId = (layerId: string, lineId: string) => `${layerId}.line_${lineId}`;
const getVectorMaskLayerRectId = (layerId: string, lineId: string) => `${layerId}.rect_${lineId}`;
export const getVectorMaskLayerObjectGroupId = (layerId: string, groupId: string) =>
  `${layerId}.objectGroup_${groupId}`;
export const getLayerBboxId = (layerId: string) => `${layerId}.bbox`;

export const regionalPromptsPersistConfig: PersistConfig<RegionalPromptsState> = {
  name: regionalPromptsSlice.name,
  initialState: initialRegionalPromptsState,
  migrate: migrateRegionalPromptsState,
  persistDenylist: [],
};

// These actions are _individually_ grouped together as single undoable actions
const undoableGroupByMatcher = isAnyOf(
  layerTranslated,
  brushSizeChanged,
  globalMaskLayerOpacityChanged,
  maskLayerPositivePromptChanged,
  maskLayerNegativePromptChanged,
  maskLayerPreviewColorChanged
);

// These are used to group actions into logical lines below (hate typos)
const LINE_1 = 'LINE_1';
const LINE_2 = 'LINE_2';

export const regionalPromptsUndoableConfig: UndoableOptions<RegionalPromptsState, UnknownAction> = {
  limit: 64,
  undoType: regionalPromptsSlice.actions.undo.type,
  redoType: regionalPromptsSlice.actions.redo.type,
  groupBy: (action, state, history) => {
    // Lines are started with `maskLayerLineAdded` and may have any number of subsequent `maskLayerPointsAdded` events.
    // We can use a double-buffer-esque trick to group each "logical" line as a single undoable action, without grouping
    // separate logical lines as a single undo action.
    if (maskLayerLineAdded.match(action)) {
      return history.group === LINE_1 ? LINE_2 : LINE_1;
    }
    if (maskLayerPointsAdded.match(action)) {
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
    if (!action.type.startsWith(regionalPromptsSlice.name)) {
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
