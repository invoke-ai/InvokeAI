import type { PayloadAction, UnknownAction } from '@reduxjs/toolkit';
import { createAction, createSlice, isAnyOf } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { moveBackward, moveForward, moveToBack, moveToFront } from 'common/util/arrayUtils';
import type { ParameterAutoNegative } from 'features/parameters/types/parameterSchemas';
import type { IRect, Vector2d } from 'konva/lib/types';
import { isEqual } from 'lodash-es';
import { atom } from 'nanostores';
import type { RgbaColor } from 'react-colorful';
import type { UndoableOptions } from 'redux-undo';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

type DrawingTool = 'brush' | 'eraser';

export type RPTool = DrawingTool | 'move';

type VectorMaskLine = {
  id: string;
  kind: 'vector_mask_line';
  tool: DrawingTool;
  strokeWidth: number;
  points: number[];
};

type VectorMaskRect = {
  id: string;
  kind: 'vector_mask_rect';
  x: number;
  y: number;
  width: number;
  height: number;
};

type TextPrompt = {
  positive: string;
  negative: string;
};

type ImagePrompt = {
  // TODO
};

type LayerBase = {
  id: string;
  x: number;
  y: number;
  bbox: IRect | null;
  bboxNeedsUpdate: boolean;
  isVisible: boolean;
};

type MaskLayerBase = LayerBase & {
  textPrompt: TextPrompt | null; // Up to one text prompt per mask
  imagePrompts: ImagePrompt[]; // Any number of image prompts
  previewColor: RgbaColor;
  autoNegative: ParameterAutoNegative;
  needsPixelBbox: boolean; // Needs the slower pixel-based bbox calculation - set to true when an there is an eraser object
};

export type VectorMaskLayer = MaskLayerBase & {
  kind: 'vector_mask_layer';
  objects: (VectorMaskLine | VectorMaskRect)[];
};

export type Layer = VectorMaskLayer;

type RegionalPromptsState = {
  _version: 1;
  selectedLayerId: string | null;
  layers: Layer[];
  brushSize: number;
  brushColor: RgbaColor;
  globalMaskLayerOpacity: number;
  isEnabled: boolean;
};

export const initialRegionalPromptsState: RegionalPromptsState = {
  _version: 1,
  selectedLayerId: null,
  brushSize: 100,
  brushColor: { r: 255, g: 0, b: 0, a: 1 },
  layers: [],
  globalMaskLayerOpacity: 0.5, // This currently doesn't work
  isEnabled: false,
};

const isLine = (obj: VectorMaskLine | VectorMaskRect): obj is VectorMaskLine => obj.kind === 'vector_mask_line';
export const isVectorMaskLayer = (layer?: Layer): layer is VectorMaskLayer => layer?.kind === 'vector_mask_layer';

export const regionalPromptsSlice = createSlice({
  name: 'regionalPrompts',
  initialState: initialRegionalPromptsState,
  reducers: {
    //#region Meta Layer
    layerAdded: {
      reducer: (state, action: PayloadAction<Layer['kind'], string, { uuid: string }>) => {
        const kind = action.payload;
        if (action.payload === 'vector_mask_layer') {
          const lastColor = state.layers[state.layers.length - 1]?.previewColor;
          const color = LayerColors.next(lastColor);
          const layer: VectorMaskLayer = {
            id: getVectorMaskLayerId(action.meta.uuid),
            kind,
            isVisible: true,
            bbox: null,
            bboxNeedsUpdate: false,
            objects: [],
            previewColor: color,
            x: 0,
            y: 0,
            autoNegative: 'off',
            needsPixelBbox: false,
            textPrompt: {
              positive: '',
              negative: '',
            },
            imagePrompts: [],
          };
          state.layers.push(layer);
          state.selectedLayerId = layer.id;
          return;
        }
      },
      prepare: (payload: Layer['kind']) => ({ payload, meta: { uuid: uuidv4() } }),
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
        layer.objects = [];
        layer.bbox = null;
        layer.isVisible = true;
        layer.needsPixelBbox = false;
        layer.bboxNeedsUpdate = false;
      }
    },
    layerDeleted: (state, action: PayloadAction<string>) => {
      state.layers = state.layers.filter((l) => l.id !== action.payload);
      state.selectedLayerId = state.layers[0]?.id ?? null;
    },
    allLayersDeleted: (state) => {
      state.layers = [];
      state.selectedLayerId = null;
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
    maskLayerPositivePromptChanged: (state, action: PayloadAction<{ layerId: string; prompt: string }>) => {
      const { layerId, prompt } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (layer && layer.textPrompt) {
        layer.textPrompt.positive = prompt;
      }
    },
    maskLayerNegativePromptChanged: (state, action: PayloadAction<{ layerId: string; prompt: string }>) => {
      const { layerId, prompt } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (layer && layer.textPrompt) {
        layer.textPrompt.negative = prompt;
      }
    },
    maskLayerPreviewColorChanged: (state, action: PayloadAction<{ layerId: string; color: RgbaColor }>) => {
      const { layerId, color } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (layer) {
        layer.previewColor = color;
      }
    },
    lineAdded: {
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
        if (layer) {
          const lineId = getVectorMaskLayerLineId(layer.id, action.meta.uuid);
          layer.objects.push({
            kind: 'vector_mask_line',
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
    pointsAddedToLastLine: (state, action: PayloadAction<{ layerId: string; point: [number, number] }>) => {
      const { layerId, point } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (layer) {
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
    maskLayerAutoNegativeChanged: (
      state,
      action: PayloadAction<{ layerId: string; autoNegative: ParameterAutoNegative }>
    ) => {
      const { layerId, autoNegative } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (layer) {
        layer.autoNegative = autoNegative;
      }
    },
    //#endregion
    //#region General
    brushSizeChanged: (state, action: PayloadAction<number>) => {
      state.brushSize = action.payload;
    },
    globalMaskLayerOpacityChanged: (state, action: PayloadAction<number>) => {
      state.globalMaskLayerOpacity = action.payload;
      for (const layer of state.layers) {
        layer.previewColor.a = action.payload;
      }
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
  static COLORS: RgbaColor[] = [
    { r: 123, g: 159, b: 237, a: 1 }, // rgb(123, 159, 237)
    { r: 106, g: 222, b: 106, a: 1 }, // rgb(106, 222, 106)
    { r: 250, g: 225, b: 80, a: 1 }, // rgb(250, 225, 80)
    { r: 233, g: 137, b: 81, a: 1 }, // rgb(233, 137, 81)
    { r: 229, g: 96, b: 96, a: 1 }, // rgb(229, 96, 96)
    { r: 226, g: 122, b: 210, a: 1 }, // rgb(226, 122, 210)
    { r: 167, g: 116, b: 234, a: 1 }, // rgb(167, 116, 234)
  ];
  static i = this.COLORS.length - 1;
  /**
   * Get the next color in the sequence. If a known color is provided, the next color will be the one after it.
   */
  static next(currentColor?: RgbaColor): RgbaColor {
    if (currentColor) {
      const i = this.COLORS.findIndex((c) => isEqual(c, { ...currentColor, a: 1 }));
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
  maskLayerAutoNegativeChanged,
  layerBboxChanged,
  maskLayerPreviewColorChanged,
  layerVisibilityToggled,
  lineAdded,
  maskLayerNegativePromptChanged,
  pointsAddedToLastLine,
  maskLayerPositivePromptChanged,
  layerReset,
  layerSelected,
  layerTranslated,
  // General actions
  isEnabledChanged,
  brushSizeChanged,
  globalMaskLayerOpacityChanged,
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
export const TOOL_PREVIEW_LAYER_ID = 'tool_preview_layer';
export const BRUSH_FILL_ID = 'brush_fill';
export const BRUSH_BORDER_INNER_ID = 'brush_border_inner';
export const BRUSH_BORDER_OUTER_ID = 'brush_border_outer';

// Names (aka classes) for Konva layers and objects
export const VECTOR_MASK_LAYER_NAME = 'vector_mask_layer';
export const VECTOR_MASK_LAYER_LINE_NAME = 'vector_mask_layer.line';
export const VECTOR_MASK_LAYER_OBJECT_GROUP_NAME = 'vector_mask_layer.object_group';
export const LAYER_BBOX_NAME = 'layer.bbox';

// Getters for non-singleton layer and object IDs
const getVectorMaskLayerId = (layerId: string) => `${VECTOR_MASK_LAYER_NAME}_${layerId}`;
const getVectorMaskLayerLineId = (layerId: string, lineId: string) => `${layerId}.line_${lineId}`;
export const getVectorMaskLayerObjectGroupId = (layerId: string, groupId: string) =>
  `${layerId}.objectGroup_${groupId}`;
export const getLayerBboxId = (layerId: string) => `${layerId}.bbox`;

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
  globalMaskLayerOpacityChanged,
  isEnabledChanged,
  maskLayerPositivePromptChanged,
  maskLayerNegativePromptChanged,
  layerTranslated,
  maskLayerPreviewColorChanged
);

const LINE_1 = 'LINE_1';
const LINE_2 = 'LINE_2';

export const regionalPromptsUndoableConfig: UndoableOptions<RegionalPromptsState, UnknownAction> = {
  limit: 64,
  undoType: undoRegionalPrompts.type,
  redoType: redoRegionalPrompts.type,
  groupBy: (action, state, history) => {
    // Lines are started with `lineAdded` and may have any number of subsequent `pointsAddedToLastLine` events.
    // We can use a double-buffer-esque trick to group each "logical" line as a single undoable action, without grouping
    // separate logical lines as a single undo action.
    if (lineAdded.match(action)) {
      return history.group === LINE_1 ? LINE_2 : LINE_1;
    }
    if (pointsAddedToLastLine.match(action)) {
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
    if (layerBboxChanged.match(action)) {
      return false;
    }
    return true;
  },
};
