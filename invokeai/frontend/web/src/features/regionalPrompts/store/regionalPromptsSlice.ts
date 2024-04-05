import type { EntityState } from '@reduxjs/toolkit';
import { createEntityAdapter, createSlice } from '@reduxjs/toolkit';
import { getSelectorsOptions } from 'app/store/createMemoizedSelector';
import type { PersistConfig, RootState } from 'app/store/store';
import type { RgbaColor } from 'react-colorful';
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
  color: RgbaColor;
};

export type FillRectObject = LayerObjectBase & {
  kind: 'fillRect';
  x: number;
  y: number;
  width: number;
  height: number;
  color: RgbaColor;
};

export type LayerObject = ImageObject | LineObject | FillRectObject;

export type PromptRegionLayer = {
  id: string;
  objects: EntityState<LayerObject, string>;
  prompt: string;
};

export const layersAdapter = createEntityAdapter<PromptRegionLayer, string>({
  selectId: (layer) => layer.id,
});
export const layersSelectors = layersAdapter.getSelectors(undefined, getSelectorsOptions);

export const layerObjectsAdapter = createEntityAdapter<LayerObject, string>({
  selectId: (obj) => obj.id,
});
export const layerObjectsSelectors = layerObjectsAdapter.getSelectors(undefined, getSelectorsOptions);

const getMockState = () => {
  // Mock data
  const layer1ID = uuidv4();
  const obj1ID = uuidv4();
  const obj2ID = uuidv4();

  const objectEntities: Record<string, LayerObject> = {
    [obj1ID]: {
      id: obj1ID,
      kind: 'line',
      isSelected: false,
      color: { r: 255, g: 0, b: 0, a: 1 },
      strokeWidth: 5,
      points: [20, 20, 100, 100],
    },
    [obj2ID]: {
      id: obj2ID,
      kind: 'fillRect',
      isSelected: false,
      color: { r: 0, g: 255, b: 0, a: 1 },
      x: 150,
      y: 150,
      width: 100,
      height: 100,
    },
  };
  const objectsInitialState = layerObjectsAdapter.getInitialState(undefined, objectEntities);
  const entities: Record<string, PromptRegionLayer> = {
    [layer1ID]: {
      id: layer1ID,
      prompt: 'strawberries',
      objects: objectsInitialState,
    },
  };

  return entities;
};

export const initialRegionalPromptsState = layersAdapter.getInitialState(
  { _version: 1, selectedID: null },
  getMockState()
);

export type RegionalPromptsState = typeof initialRegionalPromptsState;

export const regionalPromptsSlice = createSlice({
  name: 'regionalPrompts',
  initialState: initialRegionalPromptsState,
  reducers: {
    layerAdded: layersAdapter.addOne,
    layerRemoved: layersAdapter.removeOne,
    layerUpdated: layersAdapter.updateOne,
    layersReset: layersAdapter.removeAll,
  },
});

export const { layerAdded, layerRemoved, layerUpdated, layersReset } = regionalPromptsSlice.actions;

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
