import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { deepClone } from 'common/util/deepClone';
import { getFormFieldInitialValues } from 'features/nodes/store/nodesSlice';
import { SHARED_NODE_PROPERTIES } from 'features/nodes/types/constants';
import type { NodesState } from 'features/nodes/store/types';
import { zNodesState } from 'features/nodes/store/types';
import type { StatefulFieldValue } from 'features/nodes/types/field';
import type { AnyNode } from 'features/nodes/types/invocation';
import { isInvocationNode } from 'features/nodes/types/invocation';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { z } from 'zod';

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
  reducers: {
    canvasWorkflowNodesCleared: () => getInitialState(),
    // Field value mutations - these update the shadow nodes when fields are changed
    fieldStringValueChanged: (state, action: FieldValueAction<StatefulFieldValue>) => {
      fieldValueReducer(state, action, z.any());
    },
    fieldIntegerValueChanged: (state, action: FieldValueAction<StatefulFieldValue>) => {
      fieldValueReducer(state, action, z.any());
    },
    fieldFloatValueChanged: (state, action: FieldValueAction<StatefulFieldValue>) => {
      fieldValueReducer(state, action, z.any());
    },
    fieldBooleanValueChanged: (state, action: FieldValueAction<StatefulFieldValue>) => {
      fieldValueReducer(state, action, z.any());
    },
    fieldModelIdentifierValueChanged: (state, action: FieldValueAction<StatefulFieldValue>) => {
      fieldValueReducer(state, action, z.any());
    },
    fieldEnumModelValueChanged: (state, action: FieldValueAction<StatefulFieldValue>) => {
      fieldValueReducer(state, action, z.any());
    },
    fieldSchedulerValueChanged: (state, action: FieldValueAction<StatefulFieldValue>) => {
      fieldValueReducer(state, action, z.any());
    },
    fieldBoardValueChanged: (state, action: FieldValueAction<StatefulFieldValue>) => {
      fieldValueReducer(state, action, z.any());
    },
    fieldImageValueChanged: (state, action: FieldValueAction<StatefulFieldValue>) => {
      fieldValueReducer(state, action, z.any());
    },
    fieldColorValueChanged: (state, action: FieldValueAction<StatefulFieldValue>) => {
      fieldValueReducer(state, action, z.any());
    },
    fieldImageCollectionValueChanged: (state, action: FieldValueAction<StatefulFieldValue>) => {
      fieldValueReducer(state, action, z.any());
    },
    fieldStringCollectionValueChanged: (state, action: FieldValueAction<StatefulFieldValue>) => {
      fieldValueReducer(state, action, z.any());
    },
    fieldIntegerCollectionValueChanged: (state, action: FieldValueAction<StatefulFieldValue>) => {
      fieldValueReducer(state, action, z.any());
    },
    fieldFloatCollectionValueChanged: (state, action: FieldValueAction<StatefulFieldValue>) => {
      fieldValueReducer(state, action, z.any());
    },
    fieldFloatGeneratorValueChanged: (state, action: FieldValueAction<StatefulFieldValue>) => {
      fieldValueReducer(state, action, z.any());
    },
    fieldIntegerGeneratorValueChanged: (state, action: FieldValueAction<StatefulFieldValue>) => {
      fieldValueReducer(state, action, z.any());
    },
    fieldStringGeneratorValueChanged: (state, action: FieldValueAction<StatefulFieldValue>) => {
      fieldValueReducer(state, action, z.any());
    },
    fieldImageGeneratorValueChanged: (state, action: FieldValueAction<StatefulFieldValue>) => {
      fieldValueReducer(state, action, z.any());
    },
    fieldValueReset: (state, action: FieldValueAction<StatefulFieldValue>) => {
      fieldValueReducer(state, action, z.any());
    },
  },
  extraReducers(builder) {
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

          if ('data' in element && element.data && 'children' in element.data) {
            // Filter children and update the container
            const filteredChildren = element.data.children.filter(filterNodeFields);
            filteredForm.elements[elementId] = {
              ...element,
              data: {
                ...element.data,
                children: filteredChildren,
              },
            } as any;
          }

          return true;
        };

        // Start filtering from root
        const filteredChildren = rootElement.data.children.filter(filterNodeFields);
        filteredForm.elements[filteredForm.rootElementId] = {
          ...rootElement,
          data: {
            ...rootElement.data,
            children: filteredChildren,
          },
        } as any;
      }

      const formFieldInitialValues = getFormFieldInitialValues(filteredForm, nodes);

      const loadedNodes = nodes.map((node: AnyNode) => ({ ...SHARED_NODE_PROPERTIES, ...node }));
      console.log('[canvasWorkflowNodesSlice] Loading nodes:', loadedNodes.map((n: any) => ({ id: n.id, type: n.type, inputs: n.data?.inputs ? Object.keys(n.data.inputs) : [] })));

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
    builder.addCase(selectCanvasWorkflow.rejected, (state) => {
      return getInitialState();
    });
  },
});

export const {
  canvasWorkflowNodesCleared,
  fieldStringValueChanged,
  fieldIntegerValueChanged,
  fieldFloatValueChanged,
  fieldBooleanValueChanged,
  fieldModelIdentifierValueChanged,
  fieldEnumModelValueChanged,
  fieldSchedulerValueChanged,
  fieldBoardValueChanged,
  fieldImageValueChanged,
  fieldColorValueChanged,
  fieldImageCollectionValueChanged,
  fieldStringCollectionValueChanged,
  fieldIntegerCollectionValueChanged,
  fieldFloatCollectionValueChanged,
  fieldFloatGeneratorValueChanged,
  fieldIntegerGeneratorValueChanged,
  fieldStringGeneratorValueChanged,
  fieldImageGeneratorValueChanged,
  fieldValueReset,
} = slice.actions;

export const canvasWorkflowNodesSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zNodesState,
  getInitialState,
  persistConfig: {
    migrate: (state) => state as NodesState,
    // We don't persist this slice - it's derived from canvasWorkflow
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
      'nodes',
      'edges',
      'id',
    ],
  },
};

export const selectCanvasWorkflowNodesSlice = (state: RootState) => state.canvasWorkflowNodes;