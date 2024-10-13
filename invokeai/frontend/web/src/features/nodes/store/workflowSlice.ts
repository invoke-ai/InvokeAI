import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { workflowLoaded } from 'features/nodes/store/actions';
import { isAnyNodeOrEdgeMutation, nodeEditorReset, nodesChanged } from 'features/nodes/store/nodesSlice';
import type {
  FieldIdentifierWithValue,
  WorkflowMode,
  WorkflowsState as WorkflowState,
} from 'features/nodes/store/types';
import type { FieldIdentifier } from 'features/nodes/types/field';
import { isInvocationNode } from 'features/nodes/types/invocation';
import type { WorkflowCategory, WorkflowV3 } from 'features/nodes/types/workflow';
import { isEqual, omit, uniqBy } from 'lodash-es';
import type { SQLiteDirection, WorkflowRecordOrderBy } from 'services/api/types';

import { selectNodesSlice } from './selectors';

const blankWorkflow: Omit<WorkflowV3, 'nodes' | 'edges'> = {
  name: '',
  author: '',
  description: '',
  version: '',
  contact: '',
  tags: '',
  notes: '',
  exposedFields: [],
  meta: { version: '3.0.0', category: 'user' },
  id: undefined,
};

const initialWorkflowState: WorkflowState = {
  _version: 1,
  isTouched: false,
  mode: 'view',
  originalExposedFieldValues: [],
  searchTerm: '',
  orderBy: undefined, // initial value is decided in component
  orderDirection: 'DESC',
  categorySections: {},
  ...blankWorkflow,
};

export const workflowSlice = createSlice({
  name: 'workflow',
  initialState: initialWorkflowState,
  reducers: {
    workflowModeChanged: (state, action: PayloadAction<WorkflowMode>) => {
      state.mode = action.payload;
    },
    workflowSearchTermChanged: (state, action: PayloadAction<string>) => {
      state.searchTerm = action.payload;
    },
    workflowOrderByChanged: (state, action: PayloadAction<WorkflowRecordOrderBy>) => {
      state.orderBy = action.payload;
    },
    workflowOrderDirectionChanged: (state, action: PayloadAction<SQLiteDirection>) => {
      state.orderDirection = action.payload;
    },
    categorySectionsChanged: (state, action: PayloadAction<{ id: string; isOpen: boolean }>) => {
      const { id, isOpen } = action.payload;
      state.categorySections[id] = isOpen;
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
        ...deepClone(initialWorkflowState),
        ...deepClone(workflowExtra),
        originalExposedFieldValues,
        mode: state.mode,
      };
    });

    builder.addCase(nodeEditorReset, (state) => {
      const mode = state.mode;
      const newState = deepClone(initialWorkflowState);
      newState.mode = mode;
      return newState;
    });

    builder.addCase(nodesChanged, (state, action) => {
      // If a node was removed, we should remove any exposed fields that were associated with it. However, node changes
      // may remove and then add the same node back. For example, when updating a workflow, we replace old nodes with
      // updated nodes. In this case, we should not remove the exposed fields. To handle this, we find the last remove
      // and add changes for each exposed field. If the remove change comes after the add change, we remove the exposed
      // field.
      const exposedFieldsToRemove: FieldIdentifier[] = [];
      state.exposedFields.forEach((field) => {
        const removeIndex = action.payload.findLastIndex(
          (change) => change.type === 'remove' && change.id === field.nodeId
        );
        const addIndex = action.payload.findLastIndex(
          (change) => change.type === 'add' && change.item.id === field.nodeId
        );
        if (removeIndex > addIndex) {
          exposedFieldsToRemove.push({ nodeId: field.nodeId, fieldName: field.fieldName });
        }
      });

      state.exposedFields = state.exposedFields.filter(
        (field) => !exposedFieldsToRemove.some((f) => isEqual(f, field))
      );

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

      if (filteredChanges.length > 0 || exposedFieldsToRemove.length > 0) {
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
  workflowSearchTermChanged,
  workflowOrderByChanged,
  workflowOrderDirectionChanged,
  categorySectionsChanged,
} = workflowSlice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrateWorkflowState = (state: any): any => {
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

export const selectWorkflowSlice = (state: RootState) => state.workflow;
const createWorkflowSelector = <T>(selector: Selector<WorkflowState, T>) =>
  createSelector(selectWorkflowSlice, selector);

export const selectWorkflowName = createWorkflowSelector((workflow) => workflow.name);
export const selectWorkflowId = createWorkflowSelector((workflow) => workflow.id);
export const selectWorkflowMode = createWorkflowSelector((workflow) => workflow.mode);
export const selectWorkflowIsTouched = createWorkflowSelector((workflow) => workflow.isTouched);
export const selectWorkflowSearchTerm = createWorkflowSelector((workflow) => workflow.searchTerm);
export const selectWorkflowOrderBy = createWorkflowSelector((workflow) => workflow.orderBy);
export const selectWorkflowOrderDirection = createWorkflowSelector((workflow) => workflow.orderDirection);
export const selectWorkflowDescription = createWorkflowSelector((workflow) => workflow.description);

export const selectCleanEditor = createSelector([selectNodesSlice, selectWorkflowSlice], (nodes, workflow) => {
  const noNodes = !nodes.nodes.length;
  const isTouched = workflow.isTouched;
  const savedWorkflow = !!workflow.id;
  return noNodes && !isTouched && !savedWorkflow;
});
