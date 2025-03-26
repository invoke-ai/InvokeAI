import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { deepClone } from 'common/util/deepClone';
import {
  addElement,
  getElement,
  removeElement,
  reparentElement,
} from 'features/nodes/components/sidePanel/builder/form-manipulation';
import { workflowLoaded } from 'features/nodes/store/actions';
import { isAnyNodeOrEdgeMutation, nodeEditorReset, nodesChanged } from 'features/nodes/store/nodesSlice';
import type { NodesState, WorkflowMode, WorkflowsState as WorkflowState } from 'features/nodes/store/types';
import type { FieldIdentifier, StatefulFieldValue } from 'features/nodes/types/field';
import { isInvocationNode } from 'features/nodes/types/invocation';
import type {
  BuilderForm,
  ContainerElement,
  ElementId,
  FormElement,
  HeadingElement,
  NodeFieldElement,
  TextElement,
  WorkflowCategory,
  WorkflowV3,
} from 'features/nodes/types/workflow';
import {
  buildContainer,
  getDefaultForm,
  isContainerElement,
  isHeadingElement,
  isNodeFieldElement,
  isTextElement,
} from 'features/nodes/types/workflow';
import { isEqual } from 'lodash-es';
import { useMemo } from 'react';

import { selectNodesSlice } from './selectors';

type FormElementDataChangedAction<T extends FormElement> = PayloadAction<{
  id: string;
  changes: Partial<T['data']>;
}>;

const formElementDataChangedReducer = <T extends FormElement>(
  state: WorkflowState,
  action: FormElementDataChangedAction<T>,
  guard: (element: FormElement) => element is T
) => {
  const { id, changes } = action.payload;
  const element = state.form?.elements[id];
  if (!element || !guard(element)) {
    return;
  }
  element.data = { ...element.data, ...changes } as T['data'];
};

const getBlankWorkflow = (): Omit<WorkflowV3, 'nodes' | 'edges'> => {
  return {
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
    form: getDefaultForm(),
  };
};

const initialWorkflowState: WorkflowState = {
  _version: 1,
  isTouched: false,
  mode: 'view',
  formFieldInitialValues: {},
  ...getBlankWorkflow(),
};

export const workflowSlice = createSlice({
  name: 'workflow',
  initialState: initialWorkflowState,
  reducers: {
    workflowModeChanged: (state, action: PayloadAction<WorkflowMode>) => {
      state.mode = action.payload;
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
    formReset: (state) => {
      const rootElement = buildContainer('column', []);
      state.form = {
        elements: { [rootElement.id]: rootElement },
        rootElementId: rootElement.id,
      };
      state.isTouched = true;
    },
    formElementAdded: (
      state,
      action: PayloadAction<{
        element: FormElement;
        parentId: ElementId;
        index?: number;
        initialValue?: StatefulFieldValue;
      }>
    ) => {
      const { form } = state;
      const { element, parentId, index, initialValue } = action.payload;
      addElement({ form, element, parentId, index });
      if (isNodeFieldElement(element)) {
        state.formFieldInitialValues[element.id] = initialValue;
      }
      state.isTouched = true;
    },
    formElementRemoved: (state, action: PayloadAction<{ id: string }>) => {
      const { form } = state;
      const { id } = action.payload;
      removeElement({ form, id });
      delete state.formFieldInitialValues[id];
      state.isTouched = true;
    },
    formElementReparented: (state, action: PayloadAction<{ id: string; newParentId: string; index: number }>) => {
      const { form } = state;
      const { id, newParentId, index } = action.payload;
      reparentElement({ form, id, newParentId, index });
      state.isTouched = true;
    },
    formElementHeadingDataChanged: (state, action: FormElementDataChangedAction<HeadingElement>) => {
      formElementDataChangedReducer(state, action, isHeadingElement);
      state.isTouched = true;
    },
    formElementTextDataChanged: (state, action: FormElementDataChangedAction<TextElement>) => {
      formElementDataChangedReducer(state, action, isTextElement);
      state.isTouched = true;
    },
    formElementNodeFieldDataChanged: (state, action: FormElementDataChangedAction<NodeFieldElement>) => {
      formElementDataChangedReducer(state, action, isNodeFieldElement);
      state.isTouched = true;
    },
    formElementContainerDataChanged: (state, action: FormElementDataChangedAction<ContainerElement>) => {
      formElementDataChangedReducer(state, action, isContainerElement);
      state.isTouched = true;
    },
    formFieldInitialValuesChanged: (
      state,
      action: PayloadAction<{ formFieldInitialValues: WorkflowState['formFieldInitialValues'] }>
    ) => {
      const { formFieldInitialValues } = action.payload;
      state.formFieldInitialValues = formFieldInitialValues;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(workflowLoaded, (state, action): WorkflowState => {
      const { nodes, edges: _edges, ...workflowExtra } = action.payload;

      const formFieldInitialValues = getFormFieldInitialValues(workflowExtra.form, nodes);

      return {
        ...deepClone(initialWorkflowState),
        ...deepClone(workflowExtra),
        formFieldInitialValues,
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
      const fieldsToRemove: FieldIdentifier[] = [];

      state.exposedFields.forEach((field) => {
        const removeIndex = action.payload.findLastIndex(
          (change) => change.type === 'remove' && change.id === field.nodeId
        );
        const addIndex = action.payload.findLastIndex(
          (change) => change.type === 'add' && change.item.id === field.nodeId
        );
        if (removeIndex > addIndex) {
          fieldsToRemove.push({ nodeId: field.nodeId, fieldName: field.fieldName });
        }
      });
      state.exposedFields = state.exposedFields.filter((field) => !fieldsToRemove.some((f) => isEqual(f, field)));

      if (state.form) {
        for (const el of Object.values(state.form?.elements || {})) {
          if (!isNodeFieldElement(el)) {
            continue;
          }
          const { nodeId } = el.data.fieldIdentifier;
          const removeIndex = action.payload.findLastIndex(
            (change) => change.type === 'remove' && change.id === nodeId
          );
          const addIndex = action.payload.findLastIndex((change) => change.type === 'add' && change.item.id === nodeId);
          if (removeIndex > addIndex) {
            removeElement({ form: state.form, id: el.id });
          }
        }
      }

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

      if (filteredChanges.length > 0 || fieldsToRemove.length > 0) {
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
  formReset,
  formElementAdded,
  formElementRemoved,
  formElementReparented,
  formElementHeadingDataChanged,
  formElementTextDataChanged,
  formElementNodeFieldDataChanged,
  formElementContainerDataChanged,
  formFieldInitialValuesChanged,
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

// The form builder's initial values are based on the current values of the node fields in the workflow.
export const getFormFieldInitialValues = (form: BuilderForm, nodes: NodesState['nodes']) => {
  const formFieldInitialValues: Record<string, StatefulFieldValue> = {};

  for (const el of Object.values(form.elements)) {
    if (!isNodeFieldElement(el)) {
      continue;
    }
    const { nodeId, fieldName } = el.data.fieldIdentifier;

    const node = nodes.find((n) => n.id === nodeId);

    if (!isInvocationNode(node)) {
      continue;
    }

    const field = node.data.inputs[fieldName];

    if (!field) {
      continue;
    }

    formFieldInitialValues[el.id] = field.value;
  }

  return formFieldInitialValues;
};

export const selectWorkflowName = createWorkflowSelector((workflow) => workflow.name);
export const selectWorkflowId = createWorkflowSelector((workflow) => workflow.id);
export const selectWorkflowMode = createWorkflowSelector((workflow) => workflow.mode);
export const selectWorkflowIsTouched = createWorkflowSelector((workflow) => workflow.isTouched);
export const selectWorkflowDescription = createWorkflowSelector((workflow) => workflow.description);
export const selectWorkflowForm = createWorkflowSelector((workflow) => workflow.form);

export const selectCleanEditor = createSelector([selectNodesSlice, selectWorkflowSlice], (nodes, workflow) => {
  const noNodes = !nodes.nodes.length;
  const isTouched = workflow.isTouched;
  const savedWorkflow = !!workflow.id;
  return noNodes && !isTouched && !savedWorkflow;
});

export const selectFormRootElementId = createWorkflowSelector((workflow) => {
  return workflow.form.rootElementId;
});
export const selectFormRootElement = createWorkflowSelector((workflow) => {
  return getElement(workflow.form, workflow.form.rootElementId, isContainerElement);
});
export const selectIsFormEmpty = createWorkflowSelector((workflow) => {
  const rootElement = workflow.form.elements[workflow.form.rootElementId];
  if (!rootElement || !isContainerElement(rootElement)) {
    return true;
  }
  return rootElement.data.children.length === 0;
});
export const selectFormInitialValues = createWorkflowSelector((workflow) => workflow.formFieldInitialValues);
export const selectNodeFieldElements = createWorkflowSelector((workflow) =>
  Object.values(workflow.form.elements).filter(isNodeFieldElement)
);
const buildSelectElement = (id: string) => createWorkflowSelector((workflow) => workflow.form?.elements[id]);
export const useElement = (id: string): FormElement | undefined => {
  const selector = useMemo(() => buildSelectElement(id), [id]);
  const element = useAppSelector(selector);
  return element;
};
