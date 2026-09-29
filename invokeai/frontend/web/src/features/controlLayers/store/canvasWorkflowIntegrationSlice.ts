import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { isPlainObject } from 'es-toolkit';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { assert } from 'tsafe';
import z from 'zod';

const zCanvasWorkflowIntegrationState = z.object({
  _version: z.literal(1),
  isOpen: z.boolean(),
  selectedWorkflowId: z.string().nullable(),
  sourceEntityIdentifier: z
    .object({
      type: z.enum(['raster_layer', 'control_layer', 'regional_guidance', 'inpaint_mask']),
      id: z.string(),
    })
    .nullable(),
  fieldValues: z.record(z.string(), z.any()).nullable(),
  // Node settings (Use Cache / Save To Gallery) the user changed in the form preview (format: "nodeId.setting").
  // Kept apart from `fieldValues`: these are node attributes, not inputs, so they are applied to the invocation
  // itself rather than written into `node.data.inputs`.
  nodeSettingValues: z.record(z.string(), z.boolean()).nullable(),
  // Which ImageField to use for canvas image (format: "nodeId.fieldName")
  selectedImageFieldKey: z.string().nullable(),
  isProcessing: z.boolean(),
});

type CanvasWorkflowIntegrationState = z.infer<typeof zCanvasWorkflowIntegrationState>;

const getInitialState = (): CanvasWorkflowIntegrationState => ({
  _version: 1,
  isOpen: false,
  selectedWorkflowId: null,
  sourceEntityIdentifier: null,
  fieldValues: null,
  nodeSettingValues: null,
  selectedImageFieldKey: null,
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
      state.nodeSettingValues = null;
    },
    canvasWorkflowIntegrationClosed: (state) => {
      state.isOpen = false;
      state.selectedWorkflowId = null;
      state.sourceEntityIdentifier = null;
      state.fieldValues = null;
      state.nodeSettingValues = null;
      state.selectedImageFieldKey = null;
      state.isProcessing = false;
    },
    canvasWorkflowIntegrationWorkflowSelected: (state, action: PayloadAction<{ workflowId: string | null }>) => {
      state.selectedWorkflowId = action.payload.workflowId;
      // Reset field values when switching workflows
      state.fieldValues = null;
      state.nodeSettingValues = null;
      state.selectedImageFieldKey = null;
    },
    canvasWorkflowIntegrationImageFieldSelected: (state, action: PayloadAction<{ fieldKey: string | null }>) => {
      state.selectedImageFieldKey = action.payload.fieldKey;
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
    canvasWorkflowIntegrationNodeSettingChanged: (
      state,
      action: PayloadAction<{ settingKey: string; value: boolean }>
    ) => {
      if (!state.nodeSettingValues) {
        state.nodeSettingValues = {};
      }
      state.nodeSettingValues[action.payload.settingKey] = action.payload.value;
    },
    canvasWorkflowIntegrationFieldValuesReset: (state) => {
      state.fieldValues = null;
      state.nodeSettingValues = null;
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
  canvasWorkflowIntegrationImageFieldSelected,
  canvasWorkflowIntegrationFieldValueChanged,
  canvasWorkflowIntegrationNodeSettingChanged,
  canvasWorkflowIntegrationProcessingStarted,
  canvasWorkflowIntegrationProcessingCompleted,
} = slice.actions;

export const canvasWorkflowIntegrationSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zCanvasWorkflowIntegrationState,
  getInitialState,
  persistConfig: {
    migrate: (state) => {
      assert(isPlainObject(state));
      if (!('_version' in state)) {
        state._version = 1;
      }
      if (!('nodeSettingValues' in state)) {
        // Persisted before node settings could be put in a form
        state.nodeSettingValues = null;
      }
      return zCanvasWorkflowIntegrationState.parse(state);
    },
    persistDenylist: ['isOpen', 'isProcessing', 'sourceEntityIdentifier'],
  },
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
export const selectCanvasWorkflowIntegrationNodeSettingValues = createCanvasWorkflowIntegrationSelector(
  (state) => state.nodeSettingValues
);
export const selectCanvasWorkflowIntegrationSelectedImageFieldKey = createCanvasWorkflowIntegrationSelector(
  (state) => state.selectedImageFieldKey
);
export const selectCanvasWorkflowIntegrationIsProcessing = createCanvasWorkflowIntegrationSelector(
  (state) => state.isProcessing
);
