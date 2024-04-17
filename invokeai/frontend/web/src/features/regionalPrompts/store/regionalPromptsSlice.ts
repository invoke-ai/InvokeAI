import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { moveBackward, moveForward, moveToBack, moveToFront } from 'common/util/arrayUtils';
import type { Stage } from 'konva/lib/Stage';
import type { IRect, Vector2d } from 'konva/lib/types';
import { atom } from 'nanostores';
import type { RgbColor } from 'react-colorful';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

export type Tool = 'brush' | 'eraser' | 'move';

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

export type LineObject = LayerObjectBase & {
  kind: 'line';
  tool: Tool;
  strokeWidth: number;
  points: number[];
};

export type FillRectObject = LayerObjectBase & {
  kind: 'fillRect';
  x: number;
  y: number;
  width: number;
  height: number;
};

type LayerObject = ImageObject | LineObject | FillRectObject;

type LayerBase = {
  id: string;
  isVisible: boolean;
  x: number;
  y: number;
  bbox: IRect | null;
};

type PromptRegionLayer = LayerBase & {
  kind: 'promptRegionLayer';
  objects: LayerObject[];
  prompt: string;
  color: RgbColor;
};

export type Layer = PromptRegionLayer;

type RegionalPromptsState = {
  _version: 1;
  tool: Tool;
  selectedLayer: string | null;
  layers: PromptRegionLayer[];
  brushSize: number;
  promptLayerOpacity: number;
};

const initialRegionalPromptsState: RegionalPromptsState = {
  _version: 1,
  tool: 'brush',
  selectedLayer: null,
  brushSize: 40,
  layers: [],
  promptLayerOpacity: 0.5,
};

const isLine = (obj: LayerObject): obj is LineObject => obj.kind === 'line';

export const regionalPromptsSlice = createSlice({
  name: 'regionalPrompts',
  initialState: initialRegionalPromptsState,
  reducers: {
    layerAdded: {
      reducer: (state, action: PayloadAction<Layer['kind'], string, { uuid: string; color: RgbColor }>) => {
        const layer: PromptRegionLayer = {
          id: getPromptRegionLayerId(action.meta.uuid),
          isVisible: true,
          bbox: null,
          kind: action.payload,
          prompt: '',
          objects: [],
          color: action.meta.color,
          x: 0,
          y: 0,
        };
        state.layers.push(layer);
        state.selectedLayer = layer.id;
      },
      prepare: (payload: Layer['kind']) => ({ payload, meta: { uuid: uuidv4(), color: LayerColors.next() } }),
    },
    layerSelected: (state, action: PayloadAction<string>) => {
      state.selectedLayer = action.payload;
    },
    layerIsVisibleToggled: (state, action: PayloadAction<string>) => {
      const layer = state.layers.find((l) => l.id === action.payload);
      if (!layer) {
        return;
      }
      layer.isVisible = !layer.isVisible;
    },
    layerReset: (state, action: PayloadAction<string>) => {
      const layer = state.layers.find((l) => l.id === action.payload);
      if (!layer) {
        return;
      }
      layer.objects = [];
      layer.bbox = null;
      layer.isVisible = true;
      layer.prompt = '';
    },
    layerDeleted: (state, action: PayloadAction<string>) => {
      state.layers = state.layers.filter((l) => l.id !== action.payload);
      state.selectedLayer = state.layers[0]?.id ?? null;
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
    layerTranslated: (state, action: PayloadAction<{ layerId: string; x: number; y: number }>) => {
      const { layerId, x, y } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (!layer) {
        return;
      }
      layer.x = x;
      layer.y = y;
    },
    layerBboxChanged: (state, action: PayloadAction<{ layerId: string; bbox: IRect | null }>) => {
      const { layerId, bbox } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (!layer) {
        return;
      }
      layer.bbox = bbox;
    },
    allLayersDeleted: (state) => {
      state.layers = [];
      state.selectedLayer = null;
    },
    promptChanged: (state, action: PayloadAction<{ layerId: string; prompt: string }>) => {
      const { layerId, prompt } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (!layer) {
        return;
      }
      layer.prompt = prompt;
    },
    promptRegionLayerColorChanged: (state, action: PayloadAction<{ layerId: string; color: RgbColor }>) => {
      const { layerId, color } = action.payload;
      const layer = state.layers.find((l) => l.id === layerId);
      if (!layer || layer.kind !== 'promptRegionLayer') {
        return;
      }
      layer.color = color;
    },
    lineAdded: {
      reducer: (state, action: PayloadAction<[number, number, number, number], string, { uuid: string }>) => {
        const layer = state.layers.find((l) => l.id === state.selectedLayer);
        if (!layer || layer.kind !== 'promptRegionLayer') {
          return;
        }
        const lineId = getPromptRegionLayerLineId(layer.id, action.meta.uuid);
        layer.objects.push({
          kind: 'line',
          tool: state.tool,
          id: lineId,
          points: [
            action.payload[0] - layer.x,
            action.payload[1] - layer.y,
            action.payload[2] - layer.x,
            action.payload[3] - layer.y,
          ],
          strokeWidth: state.brushSize,
        });
      },
      prepare: (payload: [number, number, number, number]) => ({ payload, meta: { uuid: uuidv4() } }),
    },
    pointsAdded: (state, action: PayloadAction<[number, number]>) => {
      const layer = state.layers.find((l) => l.id === state.selectedLayer);
      if (!layer || layer.kind !== 'promptRegionLayer') {
        return;
      }
      const lastLine = layer.objects.findLast(isLine);
      if (!lastLine) {
        return;
      }
      lastLine.points.push(action.payload[0] - layer.x, action.payload[1] - layer.y);
    },
    brushSizeChanged: (state, action: PayloadAction<number>) => {
      state.brushSize = action.payload;
    },
    toolChanged: (state, action: PayloadAction<Tool>) => {
      state.tool = action.payload;
    },
    promptLayerOpacityChanged: (state, action: PayloadAction<number>) => {
      state.promptLayerOpacity = action.payload;
    },
  },
});

/**
 * This class is used to cycle through a set of colors for the prompt region layers.
 */
class LayerColors {
  static COLORS: RgbColor[] = [
    { r: 200, g: 0, b: 0 },
    { r: 0, g: 200, b: 0 },
    { r: 0, g: 0, b: 200 },
    { r: 200, g: 200, b: 0 },
    { r: 0, g: 200, b: 200 },
    { r: 200, g: 0, b: 200 },
  ];
  static i = this.COLORS.length - 1;
  /**
   * Get the next color in the sequence.
   */
  static next(): RgbColor {
    this.i = (this.i + 1) % this.COLORS.length;
    const color = this.COLORS[this.i];
    assert(color);
    return color;
  }
}

export const {
  layerAdded,
  layerSelected,
  layerReset,
  layerDeleted,
  layerIsVisibleToggled,
  promptChanged,
  lineAdded,
  pointsAdded,
  promptRegionLayerColorChanged,
  brushSizeChanged,
  layerMovedForward,
  layerMovedToFront,
  layerMovedBackward,
  layerMovedToBack,
  toolChanged,
  layerTranslated,
  layerBboxChanged,
  promptLayerOpacityChanged,
  allLayersDeleted,
} = regionalPromptsSlice.actions;

export const selectRegionalPromptsSlice = (state: RootState) => state.regionalPrompts;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrateRegionalPromptsState = (state: any): any => {
  return state;
};

export const regionalPromptsPersistConfig: PersistConfig<RegionalPromptsState> = {
  name: regionalPromptsSlice.name,
  initialState: initialRegionalPromptsState,
  migrate: migrateRegionalPromptsState,
  persistDenylist: [],
};

export const $isMouseDown = atom(false);
export const $isMouseOver = atom(false);
export const $cursorPosition = atom<Vector2d | null>(null);
export const $stage = atom<Stage | null>(null);
export const getStage = (): Stage => {
  const stage = $stage.get();
  assert(stage);
  return stage;
};
export const BRUSH_PREVIEW_LAYER_ID = 'brushPreviewLayer';
export const BRUSH_PREVIEW_FILL_ID = 'brushPreviewFill';
export const BRUSH_PREVIEW_BORDER_INNER_ID = 'brushPreviewBorderInner';
export const BRUSH_PREVIEW_BORDER_OUTER_ID = 'brushPreviewBorderOuter';
export const REGIONAL_PROMPT_LAYER_NAME = 'regionalPromptLayer';
export const REGIONAL_PROMPT_LAYER_LINE_NAME = 'regionalPromptLayerLine';
export const REGIONAL_PROMPT_LAYER_OBJECT_GROUP_NAME = 'regionalPromptLayerObjectGroup';
export const REGIONAL_PROMPT_LAYER_BBOX_NAME = 'regionalPromptLayerBbox';
export const getPromptRegionLayerId = (layerId: string) => `layer_${layerId}`;
export const getPromptRegionLayerLineId = (layerId: string, lineId: string) => `${layerId}.line_${lineId}`;
export const getPromptRegionLayerObjectGroupId = (layerId: string, groupId: string) =>
  `${layerId}.objectGroup_${groupId}`;
export const getPromptRegionLayerBboxId = (layerId: string) => `${layerId}.bbox`;
