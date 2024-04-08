import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import type Konva from 'konva';
import type { Vector2d } from 'konva/lib/types';
import { atom } from 'nanostores';
import type { RgbColor } from 'react-colorful';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

type LayerObjectBase = {
  id: string;
  isSelected: boolean;
};

export type ImageObject = LayerObjectBase & {
  kind: 'image';
  imageName: string;
  x: number;
  y: number;
  width: number;
  height: number;
};

export type LineObject = LayerObjectBase & {
  kind: 'line';
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

export type LayerObject = ImageObject | LineObject | FillRectObject;

export type PromptRegionLayer = {
  id: string;
  kind: 'promptRegionLayer';
  objects: LayerObject[];
  prompt: string;
  color: RgbColor;
};

export type Layer = PromptRegionLayer;

export type Tool = 'brush';

export type RegionalPromptsState = {
  _version: 1;
  selectedLayer: string | null;
  layers: PromptRegionLayer[];
  brushSize: number;
};

export const initialRegionalPromptsState: RegionalPromptsState = {
  _version: 1,
  selectedLayer: null,
  brushSize: 40,
  layers: [],
};

const isLine = (obj: LayerObject): obj is LineObject => obj.kind === 'line';

export const regionalPromptsSlice = createSlice({
  name: 'regionalPrompts',
  initialState: initialRegionalPromptsState,
  reducers: {
    layerAdded: {
      reducer: (state, action: PayloadAction<Layer['kind'], string, { id: string }>) => {
        const newLayer = buildLayer(action.meta.id, action.payload, state.layers.length);
        state.layers.push(newLayer);
        state.selectedLayer = newLayer.id;
      },
      prepare: (payload: Layer['kind']) => ({ payload, meta: { id: uuidv4() } }),
    },
    layerSelected: (state, action: PayloadAction<string>) => {
      state.selectedLayer = action.payload;
    },
    layerReset: (state, action: PayloadAction<string>) => {
      const layer = state.layers.find((l) => l.id === action.payload);
      if (!layer) {
        return;
      }
      layer.objects = [];
    },
    layerDeleted: (state, action: PayloadAction<string>) => {
      state.layers = state.layers.filter((l) => l.id !== action.payload);
      state.selectedLayer = state.layers[0]?.id ?? null;
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
      reducer: (state, action: PayloadAction<number[], string, { id: string }>) => {
        const selectedLayer = state.layers.find((l) => l.id === state.selectedLayer);
        if (!selectedLayer || selectedLayer.kind !== 'promptRegionLayer') {
          return;
        }
        selectedLayer.objects.push(buildLine(action.meta.id, action.payload, state.brushSize));
      },
      prepare: (payload: number[]) => ({ payload, meta: { id: uuidv4() } }),
    },
    pointsAdded: (state, action: PayloadAction<number[]>) => {
      const selectedLayer = state.layers.find((l) => l.id === state.selectedLayer);
      if (!selectedLayer || selectedLayer.kind !== 'promptRegionLayer') {
        return;
      }
      const lastLine = selectedLayer.objects.findLast(isLine);
      if (!lastLine) {
        return;
      }
      lastLine.points.push(...action.payload);
    },
    brushSizeChanged: (state, action: PayloadAction<number>) => {
      state.brushSize = action.payload;
    },
  },
});

const DEFAULT_COLORS = [
  { r: 200, g: 0, b: 0 },
  { r: 0, g: 200, b: 0 },
  { r: 0, g: 0, b: 200 },
  { r: 200, g: 200, b: 0 },
  { r: 0, g: 200, b: 200 },
  { r: 200, g: 0, b: 200 },
];

const buildLayer = (id: string, kind: Layer['kind'], layerCount: number) => {
  if (kind === 'promptRegionLayer') {
    const color = DEFAULT_COLORS[layerCount % DEFAULT_COLORS.length];
    assert(color, 'Color not found');
    return {
      id,
      kind,
      prompt: '',
      objects: [],
      color,
    };
  }
  assert(false, `Unknown layer kind: ${kind}`);
};

const buildLine = (id: string, points: number[], brushSize: number): LineObject => ({
  isSelected: false,
  kind: 'line',
  id,
  points,
  strokeWidth: brushSize,
});

export const {
  layerAdded,
  layerSelected,
  layerReset,
  layerDeleted,
  promptChanged,
  lineAdded,
  pointsAdded,
  promptRegionLayerColorChanged,
  brushSizeChanged,
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
export const $isFocused = atom(false);
export const $cursorPosition = atom<Vector2d | null>(null);
export const $tool = atom<Tool>('brush');
export const $stage = atom<Konva.Stage | null>(null);
