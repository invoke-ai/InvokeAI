import { PayloadAction, createSlice } from '@reduxjs/toolkit';
import { nodeEditorReset, nodesDeleted } from 'features/nodes/store/nodesSlice';
import { WorkflowsState as WorkflowState } from 'features/nodes/store/types';
import { FieldIdentifier } from 'features/nodes/types/field';
import { WorkflowV2 } from 'features/nodes/types/workflow';
import { cloneDeep, isEqual, uniqBy } from 'lodash-es';

export const initialWorkflowState: WorkflowState = {
  name: '',
  author: '',
  description: '',
  version: '',
  contact: '',
  tags: '',
  notes: '',
  exposedFields: [],
  meta: { version: '2.0.0', category: 'user' },
};

const workflowSlice = createSlice({
  name: 'workflow',
  initialState: initialWorkflowState,
  reducers: {
    workflowExposedFieldAdded: (
      state,
      action: PayloadAction<FieldIdentifier>
    ) => {
      state.exposedFields = uniqBy(
        state.exposedFields.concat(action.payload),
        (field) => `${field.nodeId}-${field.fieldName}`
      );
    },
    workflowExposedFieldRemoved: (
      state,
      action: PayloadAction<FieldIdentifier>
    ) => {
      state.exposedFields = state.exposedFields.filter(
        (field) => !isEqual(field, action.payload)
      );
    },
    workflowNameChanged: (state, action: PayloadAction<string>) => {
      state.name = action.payload;
    },
    workflowDescriptionChanged: (state, action: PayloadAction<string>) => {
      state.description = action.payload;
    },
    workflowTagsChanged: (state, action: PayloadAction<string>) => {
      state.tags = action.payload;
    },
    workflowAuthorChanged: (state, action: PayloadAction<string>) => {
      state.author = action.payload;
    },
    workflowNotesChanged: (state, action: PayloadAction<string>) => {
      state.notes = action.payload;
    },
    workflowVersionChanged: (state, action: PayloadAction<string>) => {
      state.version = action.payload;
    },
    workflowContactChanged: (state, action: PayloadAction<string>) => {
      state.contact = action.payload;
    },
    workflowIDChanged: (state, action: PayloadAction<string>) => {
      state.id = action.payload;
    },
    workflowLoaded: (state, action: PayloadAction<WorkflowV2>) => {
      const { nodes: _nodes, edges: _edges, ...workflow } = action.payload;
      return cloneDeep(workflow);
    },
    workflowReset: () => cloneDeep(initialWorkflowState),
  },
  extraReducers: (builder) => {
    builder.addCase(nodesDeleted, (state, action) => {
      action.payload.forEach((node) => {
        state.exposedFields = state.exposedFields.filter(
          (f) => f.nodeId !== node.id
        );
      });
    });

    builder.addCase(nodeEditorReset, () => cloneDeep(initialWorkflowState));
  },
});

export const {
  workflowExposedFieldAdded,
  workflowExposedFieldRemoved,
  workflowNameChanged,
  workflowDescriptionChanged,
  workflowTagsChanged,
  workflowAuthorChanged,
  workflowNotesChanged,
  workflowVersionChanged,
  workflowContactChanged,
  workflowIDChanged,
  workflowLoaded,
  workflowReset,
} = workflowSlice.actions;

export default workflowSlice.reducer;
