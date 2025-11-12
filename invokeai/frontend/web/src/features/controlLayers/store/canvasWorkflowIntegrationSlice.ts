import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import z from 'zod';

const zCanvasWorkflowIntegrationState = z.object({
  isOpen: z.boolean(),
  selectedWorkflowId: z.string().nullable(),
  sourceEntityIdentifier: z
    .object({
      type: z.enum(['raster_layer', 'control_layer', 'regional_guidance', 'inpaint_mask']),
      id: z.string(),
    })
    .nullable(),
  fieldValues: z.record(z.string(), z.any()).nullable(),
  isProcessing: z.boolean(),
});

export type CanvasWorkflowIntegrationState = z.infer<typeof zCanvasWorkflowIntegrationState>;

const getInitialState = (): CanvasWorkflowIntegrationState => ({
  isOpen: false,
  selectedWorkflowId: null,
  sourceEntityIdentifier: null,
  fieldValues: null,
  isProcessing: false,
});

const slice = createSlice({
  name: 'canvasWorkflowIntegration',
  initialState: getInitialState(),
  reducers: {
    canvasWorkflowIntegrationOpened: (
      state,
      action: PayloadAction<{ sourceEntityIdentifier: CanvasEntityIdentifier }>
    ) => {
      state.isOpen = true;
      state.sourceEntityIdentifier = action.payload.sourceEntityIdentifier;
      state.selectedWorkflowId = null;
      state.fieldValues = null;
    },
    canvasWorkflowIntegrationClosed: (state) => {
      state.isOpen = false;
      state.selectedWorkflowId = null;
      state.sourceEntityIdentifier = null;
      state.fieldValues = null;
      state.isProcessing = false;
    },
    canvasWorkflowIntegrationWorkflowSelected: (state, action: PayloadAction<{ workflowId: string | null }>) => {
      state.selectedWorkflowId = action.payload.workflowId;
      // Reset field values when switching workflows
      state.fieldValues = null;
    },
    canvasWorkflowIntegrationFieldValueChanged: (
      state,
      action: PayloadAction<{ fieldName: string; value: unknown }>
    ) => {
      if (!state.fieldValues) {
        state.fieldValues = {};
      }
      state.fieldValues[action.payload.fieldName] = action.payload.value;
    },
    canvasWorkflowIntegrationFieldValuesReset: (state) => {
      state.fieldValues = null;
    },
    canvasWorkflowIntegrationProcessingStarted: (state) => {
      state.isProcessing = true;
    },
    canvasWorkflowIntegrationProcessingCompleted: (state) => {
      state.isProcessing = false;
    },
  },
});

export const {
  canvasWorkflowIntegrationOpened,
  canvasWorkflowIntegrationClosed,
  canvasWorkflowIntegrationWorkflowSelected,
  canvasWorkflowIntegrationFieldValueChanged,
  canvasWorkflowIntegrationFieldValuesReset,
  canvasWorkflowIntegrationProcessingStarted,
  canvasWorkflowIntegrationProcessingCompleted,
} = slice.actions;

export const canvasWorkflowIntegrationSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zCanvasWorkflowIntegrationState,
  getInitialState,
};

const selectCanvasWorkflowIntegrationSlice = (state: RootState) => state.canvasWorkflowIntegration;
const createCanvasWorkflowIntegrationSelector = <T>(selector: Selector<CanvasWorkflowIntegrationState, T>) =>
  createSelector(selectCanvasWorkflowIntegrationSlice, selector);

export const selectCanvasWorkflowIntegrationIsOpen = createCanvasWorkflowIntegrationSelector((state) => state.isOpen);
export const selectCanvasWorkflowIntegrationSelectedWorkflowId = createCanvasWorkflowIntegrationSelector(
  (state) => state.selectedWorkflowId
);
export const selectCanvasWorkflowIntegrationSourceEntityIdentifier = createCanvasWorkflowIntegrationSelector(
  (state) => state.sourceEntityIdentifier
);
export const selectCanvasWorkflowIntegrationFieldValues = createCanvasWorkflowIntegrationSelector(
  (state) => state.fieldValues
);
export const selectCanvasWorkflowIntegrationIsProcessing = createCanvasWorkflowIntegrationSelector(
  (state) => state.isProcessing
);
