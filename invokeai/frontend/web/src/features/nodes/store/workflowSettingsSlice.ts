import type { PayloadAction } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import { SelectionMode } from '@xyflow/react';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import type { Selector } from 'react-redux';
import z from 'zod';

export const zLayeringStrategy = z.enum(['network-simplex', 'longest-path']);
type LayeringStrategy = z.infer<typeof zLayeringStrategy>;
export const zLayoutDirection = z.enum(['TB', 'LR']);
type LayoutDirection = z.infer<typeof zLayoutDirection>;
export const zNodeAlignment = z.enum(['UL', 'UR', 'DL', 'DR']);
type NodeAlignment = z.infer<typeof zNodeAlignment>;

export type WorkflowSettingsState = {
  _version: 1;
  shouldShowMinimapPanel: boolean;
  layeringStrategy: LayeringStrategy;
  nodeSpacing: number;
  layerSpacing: number;
  layoutDirection: LayoutDirection;
  shouldValidateGraph: boolean;
  shouldAnimateEdges: boolean;
  nodeAlignment: NodeAlignment;
  nodeOpacity: number;
  shouldSnapToGrid: boolean;
  shouldColorEdges: boolean;
  shouldShowEdgeLabels: boolean;
  selectionMode: SelectionMode;
};

const getInitialState = (): WorkflowSettingsState => ({
  _version: 1,
  shouldShowMinimapPanel: true,
  layeringStrategy: 'network-simplex',
  nodeSpacing: 30,
  layerSpacing: 30,
  layoutDirection: 'LR',
  nodeAlignment: 'UL',
  shouldValidateGraph: true,
  shouldAnimateEdges: true,
  shouldSnapToGrid: false,
  shouldColorEdges: true,
  shouldShowEdgeLabels: false,
  nodeOpacity: 1,
  selectionMode: SelectionMode.Partial,
});

const slice = createSlice({
  name: 'workflowSettings',
  initialState: getInitialState(),
  reducers: {
    shouldShowMinimapPanelChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldShowMinimapPanel = action.payload;
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
    nodeAlignmentChanged: (state, action: PayloadAction<NodeAlignment>) => {
      state.nodeAlignment = action.payload;
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
  layeringStrategyChanged,
  nodeSpacingChanged,
  layerSpacingChanged,
  layoutDirectionChanged,
  shouldShowEdgeLabelsChanged,
  shouldSnapToGridChanged,
  nodeAlignmentChanged,
  shouldValidateGraphChanged,
  nodeOpacityChanged,
  selectionModeChanged,
} = slice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};

export const workflowSettingsSliceConfig: SliceConfig<typeof slice> = {
  slice,
  getInitialState,
  persistConfig: {
    migrate,
  },
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

export const selectLayeringStrategy = createWorkflowSettingsSelector((s) => s.layeringStrategy);
export const selectNodeSpacing = createWorkflowSettingsSelector((s) => s.nodeSpacing);
export const selectLayerSpacing = createWorkflowSettingsSelector((s) => s.layerSpacing);
export const selectLayoutDirection = createWorkflowSettingsSelector((s) => s.layoutDirection);
export const selectNodeAlignment = createWorkflowSettingsSelector((s) => s.nodeAlignment);
