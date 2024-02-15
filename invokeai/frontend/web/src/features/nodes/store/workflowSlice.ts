import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { workflowLoaded } from 'features/nodes/store/actions';
import { isAnyNodeOrEdgeMutation, nodeEditorReset, nodesChanged, nodesDeleted } from 'features/nodes/store/nodesSlice';
import type {
  FieldIdentifierWithValue,
  WorkflowMode,
  WorkflowsState as WorkflowState,
} from 'features/nodes/store/types';
import type { FieldIdentifier } from 'features/nodes/types/field';
import { isInvocationNode } from 'features/nodes/types/invocation';
import type { WorkflowCategory, WorkflowV2 } from 'features/nodes/types/workflow';
import { cloneDeep, isEqual, omit, uniqBy } from 'lodash-es';

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
  isTouched: false,
  mode: 'view',
  originalExposedFieldValues: [],
  ...blankWorkflow,
};

export const workflowSlice = createSlice({
  name: 'workflow',
  initialState: initialWorkflowState,
  reducers: {
    workflowModeChanged: (state, action: PayloadAction<WorkflowMode>) => {
      state.mode = action.payload;
    },
    workflowExposedFieldAdded: (state, action: PayloadAction<FieldIdentifierWithValue>) => {
      state.exposedFields = uniqBy(
        state.exposedFields.concat(omit(action.payload, 'value')),
        (field) => `${field.nodeId}-${field.fieldName}`
      );
      state.originalExposedFieldValues = uniqBy(
        state.originalExposedFieldValues.concat(action.payload),
        (field) => `${field.nodeId}-${field.fieldName}`
      );
      state.isTouched = true;
    },
    workflowExposedFieldRemoved: (state, action: PayloadAction<FieldIdentifier>) => {
      state.exposedFields = state.exposedFields.filter((field) => !isEqual(field, action.payload));
      state.originalExposedFieldValues = state.originalExposedFieldValues.filter(
        (field) => !isEqual(omit(field, 'value'), action.payload)
      );
      state.isTouched = true;
    },
    workflowExposedFieldsReordered: (state, action: PayloadAction<FieldIdentifier[]>) => {
      state.exposedFields = action.payload;
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
    workflowSaved: (state) => {
      state.isTouched = false;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(workflowLoaded, (state, action) => {
      const { nodes, edges: _edges, ...workflowExtra } = action.payload;

      const originalExposedFieldValues: FieldIdentifierWithValue[] = [];

      workflowExtra.exposedFields.forEach((field) => {
        const node = nodes.find((n) => n.id === field.nodeId);

        if (!isInvocationNode(node)) {
          return;
        }

        const input = node.data.inputs[field.fieldName];

        if (!input) {
          return;
        }

        const originalExposedFieldValue = {
          nodeId: field.nodeId,
          fieldName: field.fieldName,
          value: input.value,
        };
        originalExposedFieldValues.push(originalExposedFieldValue);
      });

      return {
        ...cloneDeep(initialWorkflowState),
        ...cloneDeep(workflowExtra),
        originalExposedFieldValues,
        mode: state.mode,
      };
    });

    builder.addCase(nodesDeleted, (state, action) => {
      action.payload.forEach((node) => {
        state.exposedFields = state.exposedFields.filter((f) => f.nodeId !== node.id);
      });
    });

    builder.addCase(nodeEditorReset, () => cloneDeep(initialWorkflowState));

    builder.addCase(nodesChanged, (state, action) => {
      // Not all changes to nodes should result in the workflow being marked touched
      const filteredChanges = action.payload.filter((change) => {
        // We always want to mark the workflow as touched if a node is added, removed, or reset
        if (['add', 'remove', 'reset'].includes(change.type)) {
          return true;
        }

        // Position changes can change the position and the dragging status of the node - ignore if the change doesn't
        // affect the position
        if (change.type === 'position' && (change.position || change.positionAbsolute)) {
          return true;
        }

        // This change isn't relevant
        return false;
      });

      if (filteredChanges.length > 0) {
        state.isTouched = true;
      }
    });

    builder.addMatcher(isAnyNodeOrEdgeMutation, (state) => {
      state.isTouched = true;
    });
  },
});

export const {
  workflowModeChanged,
  workflowExposedFieldAdded,
  workflowExposedFieldRemoved,
  workflowExposedFieldsReordered,
  workflowNameChanged,
  workflowCategoryChanged,
  workflowDescriptionChanged,
  workflowTagsChanged,
  workflowAuthorChanged,
  workflowNotesChanged,
  workflowVersionChanged,
  workflowContactChanged,
  workflowIDChanged,
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
