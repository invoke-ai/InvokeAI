import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { workflowLoaded } from 'features/nodes/store/actions';
import { isAnyNodeOrEdgeMutation, nodeEditorReset, nodesDeleted } from 'features/nodes/store/nodesSlice';
import type { WorkflowsState as WorkflowState } from 'features/nodes/store/types';
import type { FieldIdentifier } from 'features/nodes/types/field';
import type { WorkflowCategory, WorkflowV2 } from 'features/nodes/types/workflow';
import { cloneDeep, isEqual, uniqBy } from 'lodash-es';

export const blankWorkflow: Omit<WorkflowV2, 'nodes' | 'edges'> = {
  name: '',
  author: '',
  description: '',
  version: '',
  contact: '',
  tags: '',
  notes: '',
  exposedFields: [],
  meta: { version: '2.0.0', category: 'user' },
  id: undefined,
};

export const initialWorkflowState: WorkflowState = {
  _version: 1,
  isTouched: true,
  ...blankWorkflow,
};

export const workflowSlice = createSlice({
  name: 'workflow',
  initialState: initialWorkflowState,
  reducers: {
    workflowExposedFieldAdded: (state, action: PayloadAction<FieldIdentifier>) => {
      state.exposedFields = uniqBy(
        state.exposedFields.concat(action.payload),
        (field) => `${field.nodeId}-${field.fieldName}`
      );
      state.isTouched = true;
    },
    workflowExposedFieldRemoved: (state, action: PayloadAction<FieldIdentifier>) => {
      state.exposedFields = state.exposedFields.filter((field) => !isEqual(field, action.payload));
      state.isTouched = true;
    },
    workflowNameChanged: (state, action: PayloadAction<string>) => {
      state.name = action.payload;
      state.isTouched = true;
    },
    workflowCategoryChanged: (state, action: PayloadAction<WorkflowCategory | undefined>) => {
      if (action.payload) {
        state.meta.category = action.payload;
      }
    },
    workflowDescriptionChanged: (state, action: PayloadAction<string>) => {
      state.description = action.payload;
      state.isTouched = true;
    },
    workflowTagsChanged: (state, action: PayloadAction<string>) => {
      state.tags = action.payload;
      state.isTouched = true;
    },
    workflowAuthorChanged: (state, action: PayloadAction<string>) => {
      state.author = action.payload;
      state.isTouched = true;
    },
    workflowNotesChanged: (state, action: PayloadAction<string>) => {
      state.notes = action.payload;
      state.isTouched = true;
    },
    workflowVersionChanged: (state, action: PayloadAction<string>) => {
      state.version = action.payload;
      state.isTouched = true;
    },
    workflowContactChanged: (state, action: PayloadAction<string>) => {
      state.contact = action.payload;
      state.isTouched = true;
    },
    workflowIDChanged: (state, action: PayloadAction<string>) => {
      state.id = action.payload;
    },
    workflowReset: () => cloneDeep(initialWorkflowState),
    workflowSaved: (state) => {
      state.isTouched = false;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(workflowLoaded, (state, action) => {
      const { nodes: _nodes, edges: _edges, ...workflowExtra } = action.payload;
      return { ...initialWorkflowState, ...cloneDeep(workflowExtra) };
    });

    builder.addCase(nodesDeleted, (state, action) => {
      action.payload.forEach((node) => {
        state.exposedFields = state.exposedFields.filter((f) => f.nodeId !== node.id);
      });
    });

    builder.addCase(nodeEditorReset, () => cloneDeep(initialWorkflowState));

    builder.addMatcher(isAnyNodeOrEdgeMutation, (state) => {
      state.isTouched = true;
    });
  },
});

export const {
  workflowExposedFieldAdded,
  workflowExposedFieldRemoved,
  workflowNameChanged,
  workflowCategoryChanged,
  workflowDescriptionChanged,
  workflowTagsChanged,
  workflowAuthorChanged,
  workflowNotesChanged,
  workflowVersionChanged,
  workflowContactChanged,
  workflowIDChanged,
  workflowReset,
  workflowSaved,
} = workflowSlice.actions;

export const selectWorkflowSlice = (state: RootState) => state.workflow;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
export const migrateWorkflowState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};

export const workflowPersistConfig: PersistConfig<WorkflowState> = {
  name: workflowSlice.name,
  initialState: initialWorkflowState,
  migrate: migrateWorkflowState,
  persistDenylist: [],
};
