import type { PayloadAction, UnknownAction } from '@reduxjs/toolkit';
import { createSlice, isAnyOf } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { deepClone } from 'common/util/deepClone';
import { isCanvasWorkflowAction } from 'features/nodes/store/actionRouter';
import { getFormFieldInitialValues } from 'features/nodes/store/nodesSlice';
import * as nodesSliceActions from 'features/nodes/store/nodesSlice';
import type { NodesState } from 'features/nodes/store/types';
import { zNodesState } from 'features/nodes/store/types';
import { SHARED_NODE_PROPERTIES } from 'features/nodes/types/constants';
import type { StatefulFieldValue } from 'features/nodes/types/field';
import {
  zBoardFieldValue,
  zBooleanFieldValue,
  zColorFieldValue,
  zEnumFieldValue,
  zFloatFieldCollectionValue,
  zFloatFieldValue,
  zFloatGeneratorFieldValue,
  zImageFieldCollectionValue,
  zImageFieldValue,
  zImageGeneratorFieldValue,
  zIntegerFieldCollectionValue,
  zIntegerFieldValue,
  zIntegerGeneratorFieldValue,
  zModelIdentifierFieldValue,
  zSchedulerFieldValue,
  zStatefulFieldValue,
  zStringFieldCollectionValue,
  zStringFieldValue,
  zStringGeneratorFieldValue,
} from 'features/nodes/types/field';
import type { AnyNode } from 'features/nodes/types/invocation';
import { isInvocationNode } from 'features/nodes/types/invocation';
import type { ContainerElement } from 'features/nodes/types/workflow';
import { isContainerElement } from 'features/nodes/types/workflow';
import type { z } from 'zod';

import { selectCanvasWorkflow } from './canvasWorkflowSlice';

/**
 * This slice holds a shadow copy of canvas workflow nodes in the same format as the nodes slice.
 * This allows the existing field components to work without modification.
 *
 * The nodes in this slice are completely separate from the workflow tab nodes.
 */

const getInitialState = (): NodesState => ({
  _version: 1,
  formFieldInitialValues: {},
  name: '',
  author: '',
  description: '',
  version: '',
  contact: '',
  tags: '',
  notes: '',
  exposedFields: [],
  meta: { version: '3.0.0', category: 'user' },
  form: {
    elements: {
      root: {
        id: 'root',
        type: 'container',
        data: {
          layout: 'column',
          children: [],
        },
      },
    },
    rootElementId: 'root',
  },
  nodes: [],
  edges: [],
  id: undefined,
});

type FieldValueAction<T extends StatefulFieldValue> = PayloadAction<{
  nodeId: string;
  fieldName: string;
  value: T;
}>;

const fieldValueReducer = <T extends StatefulFieldValue>(
  state: NodesState,
  action: FieldValueAction<T>,
  schema: z.ZodType<T>
) => {
  const { nodeId, fieldName, value } = action.payload;
  const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);
  const node = state.nodes?.[nodeIndex];
  if (!isInvocationNode(node)) {
    return;
  }
  const field = node.data?.inputs[fieldName];
  if (!field) {
    return;
  }
  const result = schema.safeParse(value);
  if (!result.success) {
    return;
  }
  field.value = result.data;
};

const slice = createSlice({
  name: 'canvasWorkflowNodes',
  initialState: getInitialState(),
  reducers: {},
  extraReducers(builder) {
    // addCase must come before addMatcher
    builder.addCase(selectCanvasWorkflow.fulfilled, (state, action) => {
      const { workflow, inputNodeId } = action.payload;
      const { nodes, edges, ...workflowExtra } = workflow;

      // Filter out form elements that reference the canvas input node
      // The input node is the canvas_composite_raster_input that will be populated by the graph builder
      const filteredForm = {
        ...workflowExtra.form,
        elements: { ...workflowExtra.form.elements },
      };

      const rootElement = filteredForm.elements[filteredForm.rootElementId];
      if (rootElement && 'data' in rootElement && rootElement.data && 'children' in rootElement.data) {
        // Recursively filter out node field elements for the canvas input node
        const filterNodeFields = (elementId: string): boolean => {
          const element = filteredForm.elements[elementId];
          if (!element) {
            return false;
          }

          if (element.type === 'node-field') {
            const nodeId = element.data.fieldIdentifier.nodeId;
            // Exclude fields from the canvas input node only
            if (nodeId === inputNodeId) {
              delete filteredForm.elements[elementId];
              return false;
            }
          }

          if (isContainerElement(element)) {
            // Filter children and update the container
            const filteredChildren = element.data.children.filter(filterNodeFields);
            const updatedElement: ContainerElement = {
              ...element,
              data: {
                ...element.data,
                children: filteredChildren,
              },
            };
            filteredForm.elements[elementId] = updatedElement;
          }

          return true;
        };

        // Start filtering from root
        if (isContainerElement(rootElement)) {
          const filteredChildren = rootElement.data.children.filter(filterNodeFields);
          const updatedRootElement: ContainerElement = {
            ...rootElement,
            data: {
              ...rootElement.data,
              children: filteredChildren,
            },
          };
          filteredForm.elements[filteredForm.rootElementId] = updatedRootElement;
        }
      }

      const formFieldInitialValues = getFormFieldInitialValues(filteredForm, nodes);

      const loadedNodes = nodes.map((node: AnyNode) => ({ ...SHARED_NODE_PROPERTIES, ...node }));

      // Load the canvas workflow into shadow nodes with filtered form
      return {
        ...getInitialState(),
        ...deepClone(workflowExtra),
        form: filteredForm,
        formFieldInitialValues,
        nodes: loadedNodes,
        edges,
      };
    });
    builder.addCase(selectCanvasWorkflow.rejected, () => {
      return getInitialState();
    });

    // Listen for field mutation actions from nodesSlice and handle them if they're for canvas workflow
    // addMatcher must come after addCase
    builder.addMatcher(
      isAnyOf(
        nodesSliceActions.fieldValueReset,
        nodesSliceActions.fieldStringValueChanged,
        nodesSliceActions.fieldStringCollectionValueChanged,
        nodesSliceActions.fieldIntegerValueChanged,
        nodesSliceActions.fieldFloatValueChanged,
        nodesSliceActions.fieldFloatCollectionValueChanged,
        nodesSliceActions.fieldIntegerCollectionValueChanged,
        nodesSliceActions.fieldBooleanValueChanged,
        nodesSliceActions.fieldBoardValueChanged,
        nodesSliceActions.fieldImageValueChanged,
        nodesSliceActions.fieldImageCollectionValueChanged,
        nodesSliceActions.fieldColorValueChanged,
        nodesSliceActions.fieldModelIdentifierValueChanged,
        nodesSliceActions.fieldEnumModelValueChanged,
        nodesSliceActions.fieldSchedulerValueChanged,
        nodesSliceActions.fieldFloatGeneratorValueChanged,
        nodesSliceActions.fieldIntegerGeneratorValueChanged,
        nodesSliceActions.fieldStringGeneratorValueChanged,
        nodesSliceActions.fieldImageGeneratorValueChanged
      ),
      (
        state,
        action: PayloadAction<{ nodeId: string; fieldName: string; value: StatefulFieldValue }> & UnknownAction
      ) => {
        // Only handle if this is a canvas workflow action
        if (!isCanvasWorkflowAction(action)) {
          return;
        }

        // Determine which schema to use based on action type
        const actionType = action.type;
        let schema;
        if (actionType.includes('String')) {
          schema = zStringFieldValue;
        } else if (actionType.includes('StringCollection')) {
          schema = zStringFieldCollectionValue;
        } else if (
          actionType.includes('Integer') &&
          !actionType.includes('Generator') &&
          !actionType.includes('Collection')
        ) {
          schema = zIntegerFieldValue;
        } else if (actionType.includes('IntegerCollection')) {
          schema = zIntegerFieldCollectionValue;
        } else if (
          actionType.includes('Float') &&
          !actionType.includes('Generator') &&
          !actionType.includes('Collection')
        ) {
          schema = zFloatFieldValue;
        } else if (actionType.includes('FloatCollection')) {
          schema = zFloatFieldCollectionValue;
        } else if (actionType.includes('Boolean')) {
          schema = zBooleanFieldValue;
        } else if (actionType.includes('Board')) {
          schema = zBoardFieldValue;
        } else if (
          actionType.includes('Image') &&
          !actionType.includes('Generator') &&
          !actionType.includes('Collection')
        ) {
          schema = zImageFieldValue;
        } else if (actionType.includes('ImageCollection')) {
          schema = zImageFieldCollectionValue;
        } else if (actionType.includes('Color')) {
          schema = zColorFieldValue;
        } else if (actionType.includes('ModelIdentifier')) {
          schema = zModelIdentifierFieldValue;
        } else if (actionType.includes('Enum')) {
          schema = zEnumFieldValue;
        } else if (actionType.includes('Scheduler')) {
          schema = zSchedulerFieldValue;
        } else if (actionType.includes('FloatGenerator')) {
          schema = zFloatGeneratorFieldValue;
        } else if (actionType.includes('IntegerGenerator')) {
          schema = zIntegerGeneratorFieldValue;
        } else if (actionType.includes('StringGenerator')) {
          schema = zStringGeneratorFieldValue;
        } else if (actionType.includes('ImageGenerator')) {
          schema = zImageGeneratorFieldValue;
        } else {
          schema = zStatefulFieldValue;
        }

        fieldValueReducer(state, action, schema);
      }
    );
  },
});

// No need to export these actions anymore - they are handled via action routing
// export const { ... } = slice.actions;

export const canvasWorkflowNodesSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zNodesState,
  getInitialState,
  persistConfig: {
    migrate: (state) => state as NodesState,
    // Only persist nodes and edges - field changes should persist, but not other workflow metadata
    persistDenylist: [
      '_version',
      'formFieldInitialValues',
      'name',
      'author',
      'description',
      'version',
      'contact',
      'tags',
      'notes',
      'exposedFields',
      'meta',
      'form',
      'id',
    ],
  },
};

export const selectCanvasWorkflowNodesSlice = (state: RootState) => state.canvasWorkflowNodes;
