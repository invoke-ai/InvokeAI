import type { PayloadAction } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import type { Selector } from 'react-redux';
import { SelectionMode } from 'reactflow';

type WorkflowSettingsState = {
  _version: 1;
  shouldShowMinimapPanel: boolean;
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
const createWorkflowSettingsSelector = <T>(selector: Selector<WorkflowSettingsState, T>) =>
  createSelector(selectWorkflowSettingsSlice, selector);
export const selectShouldSnapToGrid = createWorkflowSettingsSelector((s) => s.shouldSnapToGrid);
export const selectSelectionMode = createWorkflowSettingsSelector((s) => s.selectionMode);
export const selectShouldColorEdges = createWorkflowSettingsSelector((s) => s.shouldColorEdges);
export const selectShouldAnimateEdges = createWorkflowSettingsSelector((s) => s.shouldAnimateEdges);
export const selectShouldShowEdgeLabels = createWorkflowSettingsSelector((s) => s.shouldShowEdgeLabels);
export const selectNodeOpacity = createWorkflowSettingsSelector((s) => s.nodeOpacity);
export const selectShouldShowMinimapPanel = createWorkflowSettingsSelector((s) => s.shouldShowMinimapPanel);
export const selectShouldShouldValidateGraph = createWorkflowSettingsSelector((s) => s.shouldValidateGraph);
