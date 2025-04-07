import type { PayloadAction, UnknownAction } from '@reduxjs/toolkit';
import { createSlice, isAnyOf } from '@reduxjs/toolkit';
import type {
  EdgeChange,
  EdgeSelectionChange,
  NodeChange,
  NodeDimensionChange,
  NodePositionChange,
  NodeSelectionChange,
  Viewport,
  XYPosition,
} from '@xyflow/react';
import { applyEdgeChanges, applyNodeChanges, getConnectedEdges, getIncomers, getOutgoers } from '@xyflow/react';
import type { PersistConfig } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import {
  addElement,
  removeElement,
  reparentElement,
} from 'features/nodes/components/sidePanel/builder/form-manipulation';
import type { NodesState } from 'features/nodes/store/types';
import { SHARED_NODE_PROPERTIES } from 'features/nodes/types/constants';
import type {
  BoardFieldValue,
  BooleanFieldValue,
  CLIPEmbedModelFieldValue,
  CLIPGEmbedModelFieldValue,
  CLIPLEmbedModelFieldValue,
  ColorFieldValue,
  ControlLoRAModelFieldValue,
  ControlNetModelFieldValue,
  EnumFieldValue,
  FieldValue,
  FloatFieldValue,
  FloatGeneratorFieldValue,
  FluxReduxModelFieldValue,
  FluxVAEModelFieldValue,
  ImageFieldCollectionValue,
  ImageFieldValue,
  ImageGeneratorFieldValue,
  IntegerFieldCollectionValue,
  IntegerFieldValue,
  IntegerGeneratorFieldValue,
  IPAdapterModelFieldValue,
  LLaVAModelFieldValue,
  LoRAModelFieldValue,
  MainModelFieldValue,
  ModelIdentifierFieldValue,
  SchedulerFieldValue,
  SDXLRefinerModelFieldValue,
  SigLipModelFieldValue,
  SpandrelImageToImageModelFieldValue,
  StatefulFieldValue,
  StringFieldCollectionValue,
  StringFieldValue,
  StringGeneratorFieldValue,
  T2IAdapterModelFieldValue,
  T5EncoderModelFieldValue,
  VAEModelFieldValue,
} from 'features/nodes/types/field';
import {
  zBoardFieldValue,
  zBooleanFieldValue,
  zCLIPEmbedModelFieldValue,
  zCLIPGEmbedModelFieldValue,
  zCLIPLEmbedModelFieldValue,
  zColorFieldValue,
  zControlLoRAModelFieldValue,
  zControlNetModelFieldValue,
  zEnumFieldValue,
  zFloatFieldCollectionValue,
  zFloatFieldValue,
  zFloatGeneratorFieldValue,
  zFluxReduxModelFieldValue,
  zFluxVAEModelFieldValue,
  zImageFieldCollectionValue,
  zImageFieldValue,
  zImageGeneratorFieldValue,
  zIntegerFieldCollectionValue,
  zIntegerFieldValue,
  zIntegerGeneratorFieldValue,
  zIPAdapterModelFieldValue,
  zLLaVAModelFieldValue,
  zLoRAModelFieldValue,
  zMainModelFieldValue,
  zModelIdentifierFieldValue,
  zSchedulerFieldValue,
  zSDXLRefinerModelFieldValue,
  zSigLipModelFieldValue,
  zSpandrelImageToImageModelFieldValue,
  zStatefulFieldValue,
  zStringFieldCollectionValue,
  zStringFieldValue,
  zStringGeneratorFieldValue,
  zT2IAdapterModelFieldValue,
  zT5EncoderModelFieldValue,
  zVAEModelFieldValue,
} from 'features/nodes/types/field';
import type { AnyEdge, AnyNode } from 'features/nodes/types/invocation';
import { isInvocationNode, isNotesNode } from 'features/nodes/types/invocation';
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
  getDefaultForm,
  isContainerElement,
  isHeadingElement,
  isNodeFieldElement,
  isTextElement,
} from 'features/nodes/types/workflow';
import { atom, computed } from 'nanostores';
import type { MouseEvent } from 'react';
import type { UndoableOptions } from 'redux-undo';
import type { z } from 'zod';

import type { PendingConnection, Templates } from './types';

export const getInitialWorkflow = (): Omit<NodesState, 'mode' | 'formFieldInitialValues' | '_version'> => {
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
    form: getDefaultForm(),
    nodes: [],
    edges: [],
    // Even though this value is `undefined`, the keys _must_ be present for the presistence layer to rehydrate
    // them correctly. It uses a merge strategy that relies on the keys being present.
    id: undefined,
  };
};

const initialState: NodesState = {
  _version: 1,
  formFieldInitialValues: {},
  ...getInitialWorkflow(),
};

type FieldValueAction<T extends FieldValue> = PayloadAction<{
  nodeId: string;
  fieldName: string;
  value: T;
}>;

type FormElementDataChangedAction<T extends FormElement> = PayloadAction<{
  id: string;
  changes: Partial<T['data']>;
}>;

const formElementDataChangedReducer = <T extends FormElement>(
  state: NodesState,
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

const getField = (nodeId: string, fieldName: string, state: NodesState) => {
  const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);
  const node = state.nodes?.[nodeIndex];
  if (!isInvocationNode(node)) {
    return;
  }
  return node.data?.inputs[fieldName];
};

const fieldValueReducer = <T extends FieldValue>(
  state: NodesState,
  action: FieldValueAction<T>,
  schema: z.ZodTypeAny
) => {
  const { nodeId, fieldName, value } = action.payload;
  const field = getField(nodeId, fieldName, state);
  if (!field) {
    return;
  }
  // TODO(psyche): Do we need to do this zod validation? We already have type safety from the action payload...
  const result = schema.safeParse(value);
  if (!result.success) {
    return;
  }
  field.value = result.data;
};

export const nodesSlice = createSlice({
  name: 'nodes',
  initialState: initialState,
  reducers: {
    nodesChanged: (state, action: PayloadAction<NodeChange<AnyNode>[]>) => {
      state.nodes = applyNodeChanges<AnyNode>(action.payload, state.nodes);
      // Remove edges that are no longer valid, due to a removed or otherwise changed node
      const edgeChanges: EdgeChange<AnyEdge>[] = [];
      state.edges.forEach((e) => {
        const sourceExists = state.nodes.some((n) => n.id === e.source);
        const targetExists = state.nodes.some((n) => n.id === e.target);
        if (!(sourceExists && targetExists)) {
          edgeChanges.push({ type: 'remove', id: e.id });
        }
      });
      state.edges = applyEdgeChanges<AnyEdge>(edgeChanges, state.edges);

      // If a node was removed, we should remove any form fields that were associated with it. However, node changes
      // may remove and then add the same node back. For example, when updating a workflow, we replace old nodes with
      // updated nodes. In this case, we should not remove the form fields. To handle this, we find the last remove
      // and add changes for each exposed field. If the remove change comes after the add change, we remove the exposed
      // field.
      for (const el of Object.values(state.form.elements)) {
        if (!isNodeFieldElement(el)) {
          continue;
        }
        const { nodeId } = el.data.fieldIdentifier;
        const removeIndex = action.payload.findLastIndex((change) => change.type === 'remove' && change.id === nodeId);
        const addIndex = action.payload.findLastIndex((change) => change.type === 'add' && change.item.id === nodeId);
        if (removeIndex > addIndex) {
          removeElement({ form: state.form, id: el.id });
        }
      }
    },
    edgesChanged: (state, action: PayloadAction<EdgeChange<AnyEdge>[]>) => {
      const changes: EdgeChange<AnyEdge>[] = [];
      // We may need to massage the edge changes or otherwise handle them
      action.payload.forEach((change) => {
        if (change.type === 'remove' || change.type === 'select') {
          const edge = state.edges.find((e) => e.id === change.id);
          // If we deleted or selected a collapsed edge, we need to find its "hidden" edges and do the same to them
          if (edge && edge.type === 'collapsed') {
            const hiddenEdges = state.edges.filter((e) => e.source === edge.source && e.target === edge.target);
            if (change.type === 'remove') {
              hiddenEdges.forEach(({ id }) => {
                changes.push({ type: 'remove', id });
              });
            }
            if (change.type === 'select') {
              hiddenEdges.forEach(({ id }) => {
                changes.push({ type: 'select', id, selected: change.selected });
              });
            }
          }
        }
        if (change.type === 'add') {
          if (!change.item.type) {
            // We must add the edge type!
            change.item.type = 'default';
          }
        }
        changes.push(change);
      });
      state.edges = applyEdgeChanges(changes, state.edges);
    },
    fieldLabelChanged: (
      state,
      action: PayloadAction<{
        nodeId: string;
        fieldName: string;
        label: string;
      }>
    ) => {
      const { nodeId, fieldName, label } = action.payload;
      const node = state.nodes.find((n) => n.id === nodeId);
      if (!isInvocationNode(node)) {
        return;
      }
      const field = node.data.inputs[fieldName];
      if (!field) {
        return;
      }
      field.label = label;
    },
    nodeUseCacheChanged: (state, action: PayloadAction<{ nodeId: string; useCache: boolean }>) => {
      const { nodeId, useCache } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);

      const node = state.nodes?.[nodeIndex];

      if (!isInvocationNode(node)) {
        return;
      }
      node.data.useCache = useCache;
    },
    nodeIsIntermediateChanged: (state, action: PayloadAction<{ nodeId: string; isIntermediate: boolean }>) => {
      const { nodeId, isIntermediate } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);

      const node = state.nodes?.[nodeIndex];

      if (!isInvocationNode(node)) {
        return;
      }
      node.data.isIntermediate = isIntermediate;
    },
    nodeIsOpenChanged: (state, action: PayloadAction<{ nodeId: string; isOpen: boolean }>) => {
      const { nodeId, isOpen } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);

      const node = state.nodes?.[nodeIndex];
      if (!isInvocationNode(node) && !isNotesNode(node)) {
        return;
      }

      node.data.isOpen = isOpen;

      if (!isInvocationNode(node)) {
        return;
      }

      // edges between two closed nodes should not be visible:
      // - if the node was just opened, we need to make all its edges visible
      // - if the edge was just closed, we need to check all its edges and hide them if both nodes are closed

      const connectedEdges = getConnectedEdges([node], state.edges);

      if (isOpen) {
        // reset hidden status of all edges
        connectedEdges.forEach((edge) => {
          delete edge.hidden;
        });
        // delete dummy edges
        connectedEdges.forEach((edge) => {
          if (edge.type === 'collapsed') {
            state.edges = state.edges.filter((e) => e.id !== edge.id);
          }
        });
      } else {
        const closedIncomers = getIncomers(node, state.nodes, state.edges).filter(
          (node) => isInvocationNode(node) && node.data.isOpen === false
        );

        const closedOutgoers = getOutgoers(node, state.nodes, state.edges).filter(
          (node) => isInvocationNode(node) && node.data.isOpen === false
        );

        const collapsedEdgesToCreate: AnyEdge[] = [];

        // hide all edges
        connectedEdges.forEach((edge) => {
          if (edge.target === nodeId && closedIncomers.find((node) => node.id === edge.source)) {
            edge.hidden = true;
            const collapsedEdge = collapsedEdgesToCreate.find(
              (e) => e.source === edge.source && e.target === edge.target
            );
            if (collapsedEdge) {
              collapsedEdge.data = {
                count: (collapsedEdge.data?.count ?? 0) + 1,
              };
            } else {
              collapsedEdgesToCreate.push({
                id: `${edge.source}-${edge.target}-collapsed`,
                source: edge.source,
                target: edge.target,
                type: 'collapsed',
                data: { count: 1 },
                reconnectable: false,
                selected: edge.selected,
              });
            }
          }
          if (edge.source === nodeId && closedOutgoers.find((node) => node.id === edge.target)) {
            const collapsedEdge = collapsedEdgesToCreate.find(
              (e) => e.source === edge.source && e.target === edge.target
            );
            edge.hidden = true;
            if (collapsedEdge) {
              collapsedEdge.data = {
                count: (collapsedEdge.data?.count ?? 0) + 1,
              };
            } else {
              collapsedEdgesToCreate.push({
                id: `${edge.source}-${edge.target}-collapsed`,
                source: edge.source,
                target: edge.target,
                type: 'collapsed',
                data: { count: 1 },
                reconnectable: false,
                selected: edge.selected,
              });
            }
          }
        });
        if (collapsedEdgesToCreate.length) {
          state.edges = applyEdgeChanges<AnyEdge>(
            collapsedEdgesToCreate.map((edge) => ({ type: 'add', item: edge })),
            state.edges
          );
        }
      }
    },
    nodeLabelChanged: (state, action: PayloadAction<{ nodeId: string; label: string }>) => {
      const { nodeId, label } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);
      const node = state.nodes?.[nodeIndex];
      if (isInvocationNode(node) || isNotesNode(node)) {
        node.data.label = label;
      }
    },
    nodeNotesChanged: (state, action: PayloadAction<{ nodeId: string; notes: string }>) => {
      const { nodeId, notes } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);
      const node = state.nodes?.[nodeIndex];
      if (!isInvocationNode(node)) {
        return;
      }
      node.data.notes = notes;
    },
    fieldValueReset: (state, action: FieldValueAction<StatefulFieldValue>) => {
      fieldValueReducer(state, action, zStatefulFieldValue);
    },
    fieldStringValueChanged: (state, action: FieldValueAction<StringFieldValue>) => {
      fieldValueReducer(state, action, zStringFieldValue);
    },
    fieldStringCollectionValueChanged: (state, action: FieldValueAction<StringFieldCollectionValue>) => {
      fieldValueReducer(state, action, zStringFieldCollectionValue);
    },
    fieldIntegerValueChanged: (state, action: FieldValueAction<IntegerFieldValue>) => {
      fieldValueReducer(state, action, zIntegerFieldValue);
    },
    fieldFloatValueChanged: (state, action: FieldValueAction<FloatFieldValue>) => {
      fieldValueReducer(state, action, zFloatFieldValue);
    },
    fieldFloatCollectionValueChanged: (state, action: FieldValueAction<IntegerFieldCollectionValue>) => {
      fieldValueReducer(state, action, zFloatFieldCollectionValue);
    },
    fieldIntegerCollectionValueChanged: (state, action: FieldValueAction<IntegerFieldCollectionValue>) => {
      fieldValueReducer(state, action, zIntegerFieldCollectionValue);
    },
    fieldBooleanValueChanged: (state, action: FieldValueAction<BooleanFieldValue>) => {
      fieldValueReducer(state, action, zBooleanFieldValue);
    },
    fieldBoardValueChanged: (state, action: FieldValueAction<BoardFieldValue>) => {
      fieldValueReducer(state, action, zBoardFieldValue);
    },
    fieldImageValueChanged: (state, action: FieldValueAction<ImageFieldValue>) => {
      fieldValueReducer(state, action, zImageFieldValue);
    },
    fieldImageCollectionValueChanged: (state, action: FieldValueAction<ImageFieldCollectionValue>) => {
      fieldValueReducer(state, action, zImageFieldCollectionValue);
    },
    fieldColorValueChanged: (state, action: FieldValueAction<ColorFieldValue>) => {
      fieldValueReducer(state, action, zColorFieldValue);
    },
    fieldMainModelValueChanged: (state, action: FieldValueAction<MainModelFieldValue>) => {
      fieldValueReducer(state, action, zMainModelFieldValue);
    },
    fieldModelIdentifierValueChanged: (state, action: FieldValueAction<ModelIdentifierFieldValue>) => {
      fieldValueReducer(state, action, zModelIdentifierFieldValue);
    },
    fieldRefinerModelValueChanged: (state, action: FieldValueAction<SDXLRefinerModelFieldValue>) => {
      fieldValueReducer(state, action, zSDXLRefinerModelFieldValue);
    },
    fieldVaeModelValueChanged: (state, action: FieldValueAction<VAEModelFieldValue>) => {
      fieldValueReducer(state, action, zVAEModelFieldValue);
    },
    fieldLoRAModelValueChanged: (state, action: FieldValueAction<LoRAModelFieldValue>) => {
      fieldValueReducer(state, action, zLoRAModelFieldValue);
    },
    fieldLLaVAModelValueChanged: (state, action: FieldValueAction<LLaVAModelFieldValue>) => {
      fieldValueReducer(state, action, zLLaVAModelFieldValue);
    },
    fieldControlNetModelValueChanged: (state, action: FieldValueAction<ControlNetModelFieldValue>) => {
      fieldValueReducer(state, action, zControlNetModelFieldValue);
    },
    fieldIPAdapterModelValueChanged: (state, action: FieldValueAction<IPAdapterModelFieldValue>) => {
      fieldValueReducer(state, action, zIPAdapterModelFieldValue);
    },
    fieldT2IAdapterModelValueChanged: (state, action: FieldValueAction<T2IAdapterModelFieldValue>) => {
      fieldValueReducer(state, action, zT2IAdapterModelFieldValue);
    },
    fieldSpandrelImageToImageModelValueChanged: (
      state,
      action: FieldValueAction<SpandrelImageToImageModelFieldValue>
    ) => {
      fieldValueReducer(state, action, zSpandrelImageToImageModelFieldValue);
    },
    fieldT5EncoderValueChanged: (state, action: FieldValueAction<T5EncoderModelFieldValue>) => {
      fieldValueReducer(state, action, zT5EncoderModelFieldValue);
    },
    fieldCLIPEmbedValueChanged: (state, action: FieldValueAction<CLIPEmbedModelFieldValue>) => {
      fieldValueReducer(state, action, zCLIPEmbedModelFieldValue);
    },
    fieldCLIPLEmbedValueChanged: (state, action: FieldValueAction<CLIPLEmbedModelFieldValue>) => {
      fieldValueReducer(state, action, zCLIPLEmbedModelFieldValue);
    },
    fieldCLIPGEmbedValueChanged: (state, action: FieldValueAction<CLIPGEmbedModelFieldValue>) => {
      fieldValueReducer(state, action, zCLIPGEmbedModelFieldValue);
    },
    fieldControlLoRAModelValueChanged: (state, action: FieldValueAction<ControlLoRAModelFieldValue>) => {
      fieldValueReducer(state, action, zControlLoRAModelFieldValue);
    },
    fieldFluxVAEModelValueChanged: (state, action: FieldValueAction<FluxVAEModelFieldValue>) => {
      fieldValueReducer(state, action, zFluxVAEModelFieldValue);
    },
    fieldSigLipModelValueChanged: (state, action: FieldValueAction<SigLipModelFieldValue>) => {
      fieldValueReducer(state, action, zSigLipModelFieldValue);
    },
    fieldFluxReduxModelValueChanged: (state, action: FieldValueAction<FluxReduxModelFieldValue>) => {
      fieldValueReducer(state, action, zFluxReduxModelFieldValue);
    },
    fieldEnumModelValueChanged: (state, action: FieldValueAction<EnumFieldValue>) => {
      fieldValueReducer(state, action, zEnumFieldValue);
    },
    fieldSchedulerValueChanged: (state, action: FieldValueAction<SchedulerFieldValue>) => {
      fieldValueReducer(state, action, zSchedulerFieldValue);
    },
    fieldFloatGeneratorValueChanged: (state, action: FieldValueAction<FloatGeneratorFieldValue>) => {
      fieldValueReducer(state, action, zFloatGeneratorFieldValue);
    },
    fieldIntegerGeneratorValueChanged: (state, action: FieldValueAction<IntegerGeneratorFieldValue>) => {
      fieldValueReducer(state, action, zIntegerGeneratorFieldValue);
    },
    fieldStringGeneratorValueChanged: (state, action: FieldValueAction<StringGeneratorFieldValue>) => {
      fieldValueReducer(state, action, zStringGeneratorFieldValue);
    },
    fieldImageGeneratorValueChanged: (state, action: FieldValueAction<ImageGeneratorFieldValue>) => {
      fieldValueReducer(state, action, zImageGeneratorFieldValue);
    },
    fieldDescriptionChanged: (state, action: PayloadAction<{ nodeId: string; fieldName: string; val?: string }>) => {
      const { nodeId, fieldName, val } = action.payload;
      const field = getField(nodeId, fieldName, state);
      if (!field) {
        return;
      }
      field.description = val || '';
    },
    notesNodeValueChanged: (state, action: PayloadAction<{ nodeId: string; value: string }>) => {
      const { nodeId, value } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);
      const node = state.nodes?.[nodeIndex];
      if (!isNotesNode(node)) {
        return;
      }
      node.data.notes = value;
    },
    nodeEditorReset: () => deepClone(initialState),
    workflowNameChanged: (state, action: PayloadAction<string>) => {
      state.name = action.payload;
    },
    workflowCategoryChanged: (state, action: PayloadAction<WorkflowCategory | undefined>) => {
      if (action.payload) {
        state.meta.category = action.payload;
      }
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
    formReset: (state) => {
      state.form = getDefaultForm();
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
    },
    formElementRemoved: (state, action: PayloadAction<{ id: string }>) => {
      const { form } = state;
      const { id } = action.payload;
      removeElement({ form, id });
      delete state.formFieldInitialValues[id];
    },
    formElementReparented: (state, action: PayloadAction<{ id: string; newParentId: string; index: number }>) => {
      const { form } = state;
      const { id, newParentId, index } = action.payload;
      reparentElement({ form, id, newParentId, index });
    },
    formElementHeadingDataChanged: (state, action: FormElementDataChangedAction<HeadingElement>) => {
      formElementDataChangedReducer(state, action, isHeadingElement);
    },
    formElementTextDataChanged: (state, action: FormElementDataChangedAction<TextElement>) => {
      formElementDataChangedReducer(state, action, isTextElement);
    },
    formElementNodeFieldDataChanged: (state, action: FormElementDataChangedAction<NodeFieldElement>) => {
      formElementDataChangedReducer(state, action, isNodeFieldElement);
    },
    formElementContainerDataChanged: (state, action: FormElementDataChangedAction<ContainerElement>) => {
      formElementDataChangedReducer(state, action, isContainerElement);
    },
    formFieldInitialValuesChanged: (
      state,
      action: PayloadAction<{ formFieldInitialValues: NodesState['formFieldInitialValues'] }>
    ) => {
      const { formFieldInitialValues } = action.payload;
      state.formFieldInitialValues = formFieldInitialValues;
    },
    workflowLoaded: (state, action: PayloadAction<WorkflowV3>) => {
      const { nodes, edges, is_published: _is_published, ...workflowExtra } = action.payload;

      const formFieldInitialValues = getFormFieldInitialValues(workflowExtra.form, nodes);

      return {
        ...deepClone(initialState),
        ...deepClone(workflowExtra),
        formFieldInitialValues,
        nodes: nodes.map((node) => ({ ...SHARED_NODE_PROPERTIES, ...node })),
        edges,
      };
    },
    undo: (state) => state,
    redo: (state) => state,
  },
});

export const {
  edgesChanged,
  fieldValueReset,
  fieldBoardValueChanged,
  fieldBooleanValueChanged,
  fieldColorValueChanged,
  fieldControlNetModelValueChanged,
  fieldEnumModelValueChanged,
  fieldImageValueChanged,
  fieldImageCollectionValueChanged,
  fieldIPAdapterModelValueChanged,
  fieldT2IAdapterModelValueChanged,
  fieldSpandrelImageToImageModelValueChanged,
  fieldLabelChanged,
  fieldLoRAModelValueChanged,
  fieldLLaVAModelValueChanged,
  fieldModelIdentifierValueChanged,
  fieldMainModelValueChanged,
  fieldIntegerValueChanged,
  fieldFloatValueChanged,
  fieldFloatCollectionValueChanged,
  fieldIntegerCollectionValueChanged,
  fieldRefinerModelValueChanged,
  fieldSchedulerValueChanged,
  fieldStringValueChanged,
  fieldStringCollectionValueChanged,
  fieldVaeModelValueChanged,
  fieldT5EncoderValueChanged,
  fieldCLIPEmbedValueChanged,
  fieldCLIPLEmbedValueChanged,
  fieldCLIPGEmbedValueChanged,
  fieldControlLoRAModelValueChanged,
  fieldFluxVAEModelValueChanged,
  fieldSigLipModelValueChanged,
  fieldFluxReduxModelValueChanged,
  fieldFloatGeneratorValueChanged,
  fieldIntegerGeneratorValueChanged,
  fieldStringGeneratorValueChanged,
  fieldImageGeneratorValueChanged,
  fieldDescriptionChanged,
  nodeEditorReset,
  nodeIsIntermediateChanged,
  nodeIsOpenChanged,
  nodeLabelChanged,
  nodeNotesChanged,
  nodesChanged,
  nodeUseCacheChanged,
  notesNodeValueChanged,
  workflowNameChanged,
  workflowCategoryChanged,
  workflowDescriptionChanged,
  workflowTagsChanged,
  workflowAuthorChanged,
  workflowNotesChanged,
  workflowVersionChanged,
  workflowContactChanged,
  workflowIDChanged,
  formReset,
  formElementAdded,
  formElementRemoved,
  formElementReparented,
  formElementHeadingDataChanged,
  formElementTextDataChanged,
  formElementNodeFieldDataChanged,
  formElementContainerDataChanged,
  formFieldInitialValuesChanged,
  workflowLoaded,
  undo,
  redo,
} = nodesSlice.actions;

export const $cursorPos = atom<XYPosition | null>(null);
export const $templates = atom<Templates>({});
export const $hasTemplates = computed($templates, (templates) => Object.keys(templates).length > 0);
export const $copiedNodes = atom<AnyNode[]>([]);
export const $copiedEdges = atom<AnyEdge[]>([]);
export const $edgesToCopiedNodes = atom<AnyEdge[]>([]);
export const $pendingConnection = atom<PendingConnection | null>(null);
export const $isConnectionInProgress = computed($pendingConnection, (pendingConnection) => pendingConnection !== null);
export const $edgePendingUpdate = atom<AnyEdge | null>(null);
export const $didUpdateEdge = atom(false);
export const $lastEdgeUpdateMouseEvent = atom<MouseEvent | null>(null);

export const $viewport = atom<Viewport>({ x: 0, y: 0, zoom: 1 });
export const $addNodeCmdk = atom(false);

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrateNodesState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};

export const nodesPersistConfig: PersistConfig<NodesState> = {
  name: nodesSlice.name,
  initialState: initialState,
  migrate: migrateNodesState,
  persistDenylist: [],
};

type NodeSelectionAction = {
  type: ReturnType<typeof nodesChanged>['type'];
  payload: NodeSelectionChange[];
};

type EdgeSelectionAction = {
  type: ReturnType<typeof edgesChanged>['type'];
  payload: EdgeSelectionChange[];
};

const isNodeSelectionAction = (action: UnknownAction): action is NodeSelectionAction => {
  if (!nodesChanged.match(action)) {
    return false;
  }
  if (action.payload.every((change) => change.type === 'select')) {
    return true;
  }
  return false;
};

const isEdgeSelectionAction = (action: UnknownAction): action is EdgeSelectionAction => {
  if (!edgesChanged.match(action)) {
    return false;
  }
  if (action.payload.every((change) => change.type === 'select')) {
    return true;
  }
  return false;
};

type NodeDimensionChangeAction = {
  type: ReturnType<typeof nodesChanged>['type'];
  payload: NodeDimensionChange[];
};

const isDimensionsChangeAction = (action: UnknownAction): action is NodeDimensionChangeAction => {
  if (!nodesChanged.match(action)) {
    return false;
  }
  if (action.payload.every((change) => change.type === 'dimensions')) {
    return true;
  }
  return false;
};

type NodePositionChangeAction = {
  type: ReturnType<typeof nodesChanged>['type'];
  payload: (NodeDimensionChange | NodePositionChange)[];
};

const isPositionChangeAction = (action: UnknownAction): action is NodePositionChangeAction => {
  if (!nodesChanged.match(action)) {
    return false;
  }
  if (action.payload.every((change) => change.type === 'position')) {
    return true;
  }
  return false;
};

// Match field mutations that are high frequency and should be grouped together - for example, when a user is
// typing in a text field, we don't want to create a new undo group for every keystroke.
const isHighFrequencyFieldChangeAction = isAnyOf(
  fieldLabelChanged,
  fieldIntegerValueChanged,
  fieldFloatValueChanged,
  fieldFloatCollectionValueChanged,
  fieldIntegerCollectionValueChanged,
  fieldStringValueChanged,
  fieldStringCollectionValueChanged,
  fieldFloatGeneratorValueChanged,
  fieldIntegerGeneratorValueChanged,
  fieldStringGeneratorValueChanged,
  fieldImageGeneratorValueChanged,
  fieldDescriptionChanged
);

// Match form changes that are high frequency and should be grouped together - for example, when a user is
// typing in a text field, we don't want to create a new undo group for every keystroke.
const isHighFrequencyFormChangeAction = isAnyOf(
  formElementHeadingDataChanged,
  formElementTextDataChanged,
  formElementNodeFieldDataChanged,
  formElementContainerDataChanged
);

// Match workflow changes that are high frequency and should be grouped together - for example, when a user is
// updating the workflow description, we don't want to create a new undo group for every keystroke.
const isHighFrequencyWorkflowDetailsAction = isAnyOf(
  workflowNameChanged,
  workflowDescriptionChanged,
  workflowTagsChanged,
  workflowAuthorChanged,
  workflowNotesChanged,
  workflowVersionChanged,
  workflowContactChanged
);

// Match node-scoped actions that are high frequency and should be grouped together - for example, when a user is
// updating the node label, we don't want to create a new undo group for every keystroke. Or when a user is writing
// a note in a notes node, we don't want to create a new undo group for every keystroke.
const isHighFrequencyNodeScopedAction = isAnyOf(nodeLabelChanged, nodeNotesChanged, notesNodeValueChanged);

export const nodesUndoableConfig: UndoableOptions<NodesState, UnknownAction> = {
  limit: 64,
  undoType: nodesSlice.actions.undo.type,
  redoType: nodesSlice.actions.redo.type,
  groupBy: (action, _state, _history) => {
    if (isHighFrequencyFieldChangeAction(action)) {
      // Group by type, node id and field name
      const { type, payload } = action;
      const { nodeId, fieldName } = payload;
      return `${type}-${nodeId}-${fieldName}`;
    }
    if (isPositionChangeAction(action)) {
      const ids = action.payload.map((change) => change.id).join(',');
      // Group by type and node ids
      return `dimensions-or-position-${ids}`;
    }
    if (isHighFrequencyFormChangeAction(action)) {
      // Group by type and form element id
      const { type, payload } = action;
      const { id } = payload;
      return `${type}-${id}`;
    }
    if (isHighFrequencyWorkflowDetailsAction(action)) {
      return 'workflow-details';
    }
    if (isHighFrequencyNodeScopedAction(action)) {
      const { type, payload } = action;
      const { nodeId } = payload;
      // Group by type and node id
      return `${type}-${nodeId}`;
    }
    return null;
  },
  filter: (action, _state, _history) => {
    // Ignore all actions from other slices
    if (!action.type.startsWith(nodesSlice.name)) {
      return false;
    }
    // Ignore actions that only select or deselect nodes and edges
    if (isNodeSelectionAction(action) || isEdgeSelectionAction(action)) {
      return false;
    }
    if (isDimensionsChangeAction(action)) {
      // Ignore actions that only change the dimensions of nodes - these are internal to reactflow
      return false;
    }
    return true;
  },
};

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
