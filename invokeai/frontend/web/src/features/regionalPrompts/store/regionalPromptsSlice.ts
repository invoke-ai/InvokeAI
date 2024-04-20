import type { PayloadAction, UnknownAction } from '@reduxjs/toolkit';
import { createAction, createSlice, isAnyOf } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { moveBackward, moveForward, moveToBack, moveToFront } from 'common/util/arrayUtils';
import type { ParameterAutoNegative } from 'features/parameters/types/parameterSchemas';
import type { IRect, Vector2d } from 'konva/lib/types';
import { isEqual } from 'lodash-es';
import { atom } from 'nanostores';
import type { RgbColor } from 'react-colorful';
import type { UndoableOptions } from 'redux-undo';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

type DrawingTool = 'brush' | 'eraser';

export type RPTool = DrawingTool | 'move';

type LayerObjectBase = {
  id: string;
};

type ImageObject = LayerObjectBase & {
  kind: 'image';
  imageName: string;
  x: number;
  y: number;
  width: number;
  height: number;
};

type LineObject = LayerObjectBase & {
  kind: 'line';
  tool: DrawingTool;
  strokeWidth: number;
  points: number[];
};

type FillRectObject = LayerObjectBase & {
  kind: 'fillRect';
  x: number;
  y: number;
  width: number;
  height: number;
};

type LayerObject = ImageObject | LineObject | FillRectObject;

type LayerBase = {
  id: string;
};

export type RegionalPromptLayer = LayerBase & {
  isVisible: boolean;
  x: number;
  y: number;
  bbox: IRect | null;
  bboxNeedsUpdate: boolean;
  hasEraserStrokes: boolean;
  kind: 'regionalPromptLayer';
  objects: LayerObject[];
  positivePrompt: string;
  negativePrompt: string;
  color: RgbColor;
  autoNegative: ParameterAutoNegative;
};

export type Layer = RegionalPromptLayer;

type RegionalPromptsState = {
  _version: 1;
  selectedLayerId: string | null;
  layers: Layer[];
  brushSize: number;
  promptLayerOpacity: number;
  isEnabled: boolean;
};

export const initialRegionalPromptsState: RegionalPromptsState = {
  _version: 1,
  selectedLayerId: null,
  brushSize: 100,
  layers: [],
  promptLayerOpacity: 0.5, // This currently doesn't work
  isEnabled: false,
};

const isLine = (obj: LayerObject): obj is LineObject => obj.kind === 'line';
export const isRPLayer = (layer?: Layer): layer is RegionalPromptLayer => layer?.kind === 'regionalPromptLayer';

export const regionalPromptsSlice = createSlice({
  name: 'regionalPrompts',
  initialState: initialRegionalPromptsState,
  reducers: {
    //#region Meta Layer
    layerAdded: {
      reducer: (state, action: PayloadAction<Layer['kind'], string, { uuid: string }>) => {
        if (action.payload === 'regionalPromptLayer') {
          const lastColor = state.layers[state.layers.length - 1]?.color;
          const color = LayerColors.next(lastColor);
          const layer: RegionalPromptLayer = {
            id: getRPLayerId(action.meta.uuid),
            isVisible: true,
            bbox: null,
            kind: action.payload,
            positivePrompt: '',
            negativePrompt: '',
            objects: [],
            color,
            x: 0,
            y: 0,
            autoNegative: 'off',
            bboxNeedsUpdate: false,
            hasEraserStrokes: false,
          };
          state.layers.push(layer);
          state.selectedLayerId = layer.id;
          return;
        }
      },
      prepare: (payload: Layer['kind']) => ({ payload, meta: { uuid: uuidv4() } }),
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
    //#endregion
    //#region RP Layers
    rpLayerSelected: (state, action: PayloadAction<string>) => {
      const layer = state.layers.find((l) => l.id === action.payload);
      if (isRPLayer(layer)) {
        state.selectedLayerId = layer.id;
      }
    },
    rpLayerIsVisibleToggled: (state, action: PayloadAction<string>) => {
      const layer = state.layers.find((l) => l.id === action.payload);
      if (isRPLayer(layer)) {
        layer.isVisible = !layer.isVisible;
      }
    },
    rpLayerReset: (state, action: PayloadAction<string>) => {
      const layer = state.layers.find((l) => l.id === action.payload);
      if (isRPLayer(layer)) {
        layer.objects = [];
        layer.bbox = null;
        layer.isVisible = true;
        layer.hasEraserStrokes = false;
        layer.bboxNeedsUpdate = false;
      }
    },
    rpLayerTranslated: (state, action: PayloadAction<{ layerId: string; x: number; y: number }>) => {
      const { layerId, x, y } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (isRPLayer(layer)) {
        layer.x = x;
        layer.y = y;
      }
    },
    rpLayerBboxChanged: (state, action: PayloadAction<{ layerId: string; bbox: IRect | null }>) => {
      const { layerId, bbox } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (isRPLayer(layer)) {
        layer.bbox = bbox;
        layer.bboxNeedsUpdate = false;
      }
    },
    allLayersDeleted: (state) => {
      state.layers = [];
      state.selectedLayerId = null;
    },
    rpLayerPositivePromptChanged: (state, action: PayloadAction<{ layerId: string; prompt: string }>) => {
      const { layerId, prompt } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (isRPLayer(layer)) {
        layer.positivePrompt = prompt;
      }
    },
    rpLayerNegativePromptChanged: (state, action: PayloadAction<{ layerId: string; prompt: string }>) => {
      const { layerId, prompt } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (isRPLayer(layer)) {
        layer.negativePrompt = prompt;
      }
    },
    rpLayerColorChanged: (state, action: PayloadAction<{ layerId: string; color: RgbColor }>) => {
      const { layerId, color } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (isRPLayer(layer)) {
        layer.color = color;
      }
    },
    rpLayerLineAdded: {
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
        if (isRPLayer(layer)) {
          const lineId = getRPLayerLineId(layer.id, action.meta.uuid);
          layer.objects.push({
            kind: 'line',
            tool: tool,
            id: lineId,
            // Points must be offset by the layer's x and y coordinates
            // TODO: Handle this in the event listener
            points: [points[0] - layer.x, points[1] - layer.y, points[2] - layer.x, points[3] - layer.y],
            strokeWidth: state.brushSize,
          });
          layer.bboxNeedsUpdate = true;
          if (!layer.hasEraserStrokes) {
            layer.hasEraserStrokes = tool === 'eraser';
          }
        }
      },
      prepare: (payload: { layerId: string; points: [number, number, number, number]; tool: DrawingTool }) => ({
        payload,
        meta: { uuid: uuidv4() },
      }),
    },
    rpLayerPointsAdded: (state, action: PayloadAction<{ layerId: string; point: [number, number] }>) => {
      const { layerId, point } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (isRPLayer(layer)) {
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
    rpLayerAutoNegativeChanged: (
      state,
      action: PayloadAction<{ layerId: string; autoNegative: ParameterAutoNegative }>
    ) => {
      const { layerId, autoNegative } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (isRPLayer(layer)) {
        layer.autoNegative = autoNegative;
      }
    },
    //#endregion
    //#region General
    brushSizeChanged: (state, action: PayloadAction<number>) => {
      state.brushSize = action.payload;
    },
    promptLayerOpacityChanged: (state, action: PayloadAction<number>) => {
      state.promptLayerOpacity = action.payload;
    },
    isEnabledChanged: (state, action: PayloadAction<boolean>) => {
      state.isEnabled = action.payload;
    },
    //#endregion
  },
});

/**
 * This class is used to cycle through a set of colors for the prompt region layers.
 */
class LayerColors {
  static COLORS: RgbColor[] = [
    { r: 123, g: 159, b: 237 }, // rgb(123, 159, 237)
    { r: 106, g: 222, b: 106 }, // rgb(106, 222, 106)
    { r: 250, g: 225, b: 80 }, // rgb(250, 225, 80)
    { r: 233, g: 137, b: 81 }, // rgb(233, 137, 81)
    { r: 229, g: 96, b: 96 }, // rgb(229, 96, 96)
    { r: 226, g: 122, b: 210 }, // rgb(226, 122, 210)
    { r: 167, g: 116, b: 234 }, // rgb(167, 116, 234)
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
  // Meta layer actions
  layerAdded,
  layerDeleted,
  layerMovedBackward,
  layerMovedForward,
  layerMovedToBack,
  layerMovedToFront,
  allLayersDeleted,
  // Regional Prompt layer actions
  rpLayerAutoNegativeChanged,
  rpLayerBboxChanged,
  rpLayerColorChanged,
  rpLayerIsVisibleToggled,
  rpLayerLineAdded,
  rpLayerNegativePromptChanged,
  rpLayerPointsAdded,
  rpLayerPositivePromptChanged,
  rpLayerReset,
  rpLayerSelected,
  rpLayerTranslated,
  // General actions
  isEnabledChanged,
  brushSizeChanged,
  promptLayerOpacityChanged,
} = regionalPromptsSlice.actions;

export const selectRegionalPromptsSlice = (state: RootState) => state.regionalPrompts;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrateRegionalPromptsState = (state: any): any => {
  return state;
};

export const $isMouseDown = atom(false);
export const $isMouseOver = atom(false);
export const $tool = atom<RPTool>('brush');
export const $cursorPosition = atom<Vector2d | null>(null);

// IDs for singleton layers and objects
export const BRUSH_PREVIEW_LAYER_ID = 'brushPreviewLayer';
export const BRUSH_PREVIEW_FILL_ID = 'brushPreviewFill';
export const BRUSH_PREVIEW_BORDER_INNER_ID = 'brushPreviewBorderInner';
export const BRUSH_PREVIEW_BORDER_OUTER_ID = 'brushPreviewBorderOuter';

// Names (aka classes) for Konva layers and objects
export const REGIONAL_PROMPT_LAYER_NAME = 'regionalPromptLayer';
export const REGIONAL_PROMPT_LAYER_LINE_NAME = 'regionalPromptLayerLine';
export const REGIONAL_PROMPT_LAYER_OBJECT_GROUP_NAME = 'regionalPromptLayerObjectGroup';
export const REGIONAL_PROMPT_LAYER_BBOX_NAME = 'regionalPromptLayerBbox';

// Getters for non-singleton layer and object IDs
const getRPLayerId = (layerId: string) => `rp_layer_${layerId}`;
const getRPLayerLineId = (layerId: string, lineId: string) => `${layerId}.line_${lineId}`;
export const getRPLayerObjectGroupId = (layerId: string, groupId: string) => `${layerId}.objectGroup_${groupId}`;
export const getPRLayerBboxId = (layerId: string) => `${layerId}.bbox`;

export const regionalPromptsPersistConfig: PersistConfig<RegionalPromptsState> = {
  name: regionalPromptsSlice.name,
  initialState: initialRegionalPromptsState,
  migrate: migrateRegionalPromptsState,
  persistDenylist: [],
};

// Payload-less actions for `redux-undo`
export const undoRegionalPrompts = createAction(`${regionalPromptsSlice.name}/undo`);
export const redoRegionalPrompts = createAction(`${regionalPromptsSlice.name}/redo`);

// These actions are _individually_ grouped together as single undoable actions
const undoableGroupByMatcher = isAnyOf(
  brushSizeChanged,
  promptLayerOpacityChanged,
  isEnabledChanged,
  rpLayerPositivePromptChanged,
  rpLayerNegativePromptChanged,
  rpLayerTranslated,
  rpLayerColorChanged
);

const LINE_1 = 'LINE_1';
const LINE_2 = 'LINE_2';

export const regionalPromptsUndoableConfig: UndoableOptions<RegionalPromptsState, UnknownAction> = {
  limit: 64,
  undoType: undoRegionalPrompts.type,
  redoType: redoRegionalPrompts.type,
  groupBy: (action, state, history) => {
    // Lines are started with `rpLayerLineAdded` and may have any number of subsequent `rpLayerPointsAdded` events.
    // We can use a double-buffer-esque trick to group each "logical" line as a single undoable action, without grouping
    // separate logical lines as a single undo action.
    if (rpLayerLineAdded.match(action)) {
      return history.group === LINE_1 ? LINE_2 : LINE_1;
    }
    if (rpLayerPointsAdded.match(action)) {
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
    if (!action.type.startsWith('regionalPrompts/')) {
      return false;
    }
    // This action is triggered on state changes, including when we undo. If we do not ignore this action, when we
    // undo, this action triggers and empties the future states array. Therefore, we must ignore this action.
    if (rpLayerBboxChanged.match(action)) {
      return false;
    }
    return true;
  },
};
