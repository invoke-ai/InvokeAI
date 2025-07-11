import type { PayloadAction } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import { SelectionMode } from '@xyflow/react';
import type { PersistConfig, RootState } from 'app/store/store';

export type NodePlacementStrategy = 'NETWORK_SIMPLEX' | 'BRANDES_KOEPF' | 'LINEAR_SEGMENTS' | 'SIMPLE';

export type LayeringStrategy = 'NETWORK_SIMPLEX' | 'LONGEST_PATH' | 'COFFMAN_GRAHAM';

export type LayoutDirection = 'DOWN' | 'RIGHT';

export type WorkflowSettingsState = {
  _version: 1;
  shouldShowMinimapPanel: boolean;
  nodePlacementStrategy: NodePlacementStrategy;
  layeringStrategy: LayeringStrategy;
  nodeSpacing: number;
  layerSpacing: number;
  layoutDirection: LayoutDirection;
  shouldValidateGraph: boolean;
  shouldAnimateEdges: boolean;
  nodeOpacity: number;
  shouldSnapToGrid: boolean;
  shouldColorEdges: boolean;
  shouldShowEdgeLabels: boolean;
  selectionMode: SelectionMode;
};

const initialState: WorkflowSettingsState = {
  _version: 1,
  shouldShowMinimapPanel: true,
  nodePlacementStrategy: 'NETWORK_SIMPLEX',
  layeringStrategy: 'NETWORK_SIMPLEX',
  nodeSpacing: 50,
  layerSpacing: 50,
  layoutDirection: 'RIGHT',
  shouldValidateGraph: true,
  shouldAnimateEdges: true,
  shouldSnapToGrid: false,
  shouldColorEdges: true,
  shouldShowEdgeLabels: false,
  nodeOpacity: 1,
  selectionMode: SelectionMode.Partial,
};

export const workflowSettingsSlice = createSlice({
  name: 'workflowSettings',
  initialState,
  reducers: {
    shouldShowMinimapPanelChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldShowMinimapPanel = action.payload;
    },
    nodePlacementStrategyChanged: (state, action: PayloadAction<NodePlacementStrategy>) => {
      state.nodePlacementStrategy = action.payload;
    },
    layeringStrategyChanged: (state, action: PayloadAction<LayeringStrategy>) => {
      state.layeringStrategy = action.payload;
    },
    nodeSpacingChanged: (state, action: PayloadAction<number>) => {
      state.nodeSpacing = action.payload;
    },
    layerSpacingChanged: (state, action: PayloadAction<number>) => {
      state.layerSpacing = action.payload;
    },
    layoutDirectionChanged: (state, action: PayloadAction<LayoutDirection>) => {
      state.layoutDirection = action.payload;
    },
    shouldValidateGraphChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldValidateGraph = action.payload;
    },
    shouldAnimateEdgesChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldAnimateEdges = action.payload;
    },
    shouldShowEdgeLabelsChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldShowEdgeLabels = action.payload;
    },
    shouldSnapToGridChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldSnapToGrid = action.payload;
    },
    shouldColorEdgesChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldColorEdges = action.payload;
    },
    nodeOpacityChanged: (state, action: PayloadAction<number>) => {
      state.nodeOpacity = action.payload;
    },
    selectionModeChanged: (state, action: PayloadAction<boolean>) => {
      state.selectionMode = action.payload ? SelectionMode.Full : SelectionMode.Partial;
    },
  },
});

export const {
  shouldAnimateEdgesChanged,
  shouldColorEdgesChanged,
  shouldShowMinimapPanelChanged,
  nodePlacementStrategyChanged,
  layeringStrategyChanged,
  nodeSpacingChanged,
  layerSpacingChanged,
  layoutDirectionChanged,
  shouldShowEdgeLabelsChanged,
  shouldSnapToGridChanged,
  shouldValidateGraphChanged,
  nodeOpacityChanged,
  selectionModeChanged,
} = workflowSettingsSlice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrateWorkflowSettingsState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};

export const workflowSettingsPersistConfig: PersistConfig<WorkflowSettingsState> = {
  name: workflowSettingsSlice.name,
  initialState,
  migrate: migrateWorkflowSettingsState,
  persistDenylist: [],
};

export const selectWorkflowSettingsSlice = (state: RootState) => state.workflowSettings;
const createWorkflowSettingsSelector = <T>(selector: (state: WorkflowSettingsState) => T) =>
  createSelector(selectWorkflowSettingsSlice, selector);
export const selectShouldSnapToGrid = createWorkflowSettingsSelector((s) => s.shouldSnapToGrid);
export const selectSelectionMode = createWorkflowSettingsSelector((s) => s.selectionMode);
export const selectShouldColorEdges = createWorkflowSettingsSelector((s) => s.shouldColorEdges);
export const selectShouldAnimateEdges = createWorkflowSettingsSelector((s) => s.shouldAnimateEdges);
export const selectShouldShowEdgeLabels = createWorkflowSettingsSelector((s) => s.shouldShowEdgeLabels);
export const selectNodeOpacity = createWorkflowSettingsSelector((s) => s.nodeOpacity);
export const selectShouldShowMinimapPanel = createWorkflowSettingsSelector((s) => s.shouldShowMinimapPanel);
export const selectShouldShouldValidateGraph = createWorkflowSettingsSelector((s) => s.shouldValidateGraph);

export const selectNodePlacementStrategy = createWorkflowSettingsSelector((s) => s.nodePlacementStrategy);
export const selectLayeringStrategy = createWorkflowSettingsSelector((s) => s.layeringStrategy);
export const selectNodeSpacing = createWorkflowSettingsSelector((s) => s.nodeSpacing);
export const selectLayerSpacing = createWorkflowSettingsSelector((s) => s.layerSpacing);
export const selectLayoutDirection = createWorkflowSettingsSelector((s) => s.layoutDirection);
