import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { cloneDeep, forEach, isEqual, map, uniqBy } from 'lodash-es';
import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  Connection,
  Edge,
  EdgeChange,
  EdgeRemoveChange,
  getConnectedEdges,
  getIncomers,
  getOutgoers,
  Node,
  NodeChange,
  OnConnectStartParams,
  SelectionMode,
  Viewport,
} from 'reactflow';
import { receivedOpenAPISchema } from 'services/api/thunks/schema';
import { sessionCanceled, sessionInvoked } from 'services/api/thunks/session';
import { ImageField } from 'services/api/types';
import {
  appSocketGeneratorProgress,
  appSocketInvocationComplete,
  appSocketInvocationError,
  appSocketInvocationStarted,
} from 'services/events/actions';
import { v4 as uuidv4 } from 'uuid';
import { DRAG_HANDLE_CLASSNAME } from '../types/constants';
import {
  BooleanInputFieldValue,
  ColorInputFieldValue,
  ControlNetModelInputFieldValue,
  CurrentImageNodeData,
  EnumInputFieldValue,
  FieldIdentifier,
  FloatInputFieldValue,
  ImageInputFieldValue,
  InputFieldValue,
  IntegerInputFieldValue,
  InvocationNodeData,
  InvocationTemplate,
  isInvocationNode,
  isNotesNode,
  LoRAModelInputFieldValue,
  MainModelInputFieldValue,
  NodeExecutionState,
  NodeStatus,
  NotesNodeData,
  SchedulerInputFieldValue,
  SDXLRefinerModelInputFieldValue,
  StringInputFieldValue,
  VaeModelInputFieldValue,
  Workflow,
} from '../types/types';
import { NodesState } from './types';
import { findUnoccupiedPosition } from './util/findUnoccupiedPosition';

export const WORKFLOW_FORMAT_VERSION = '1.0.0';

const initialNodeExecutionState: Omit<NodeExecutionState, 'nodeId'> = {
  status: NodeStatus.PENDING,
  error: null,
  progress: null,
  progressImage: null,
  outputs: [],
};

export const initialWorkflow = {
  meta: {
    version: WORKFLOW_FORMAT_VERSION,
  },
  name: '',
  author: '',
  description: '',
  notes: '',
  tags: '',
  contact: '',
  version: '',
  exposedFields: [],
};

export const initialNodesState: NodesState = {
  nodes: [],
  edges: [],
  nodeTemplates: {},
  isReady: false,
  connectionStartParams: null,
  currentConnectionFieldType: null,
  shouldShowFieldTypeLegend: false,
  shouldShowMinimapPanel: true,
  shouldValidateGraph: true,
  shouldAnimateEdges: true,
  shouldSnapToGrid: false,
  shouldColorEdges: true,
  isAddNodePopoverOpen: false,
  nodeOpacity: 1,
  selectedNodes: [],
  selectedEdges: [],
  workflow: initialWorkflow,
  nodeExecutionStates: {},
  viewport: { x: 0, y: 0, zoom: 1 },
  mouseOverField: null,
  mouseOverNode: null,
  nodesToCopy: [],
  edgesToCopy: [],
  selectionMode: SelectionMode.Partial,
};

type FieldValueAction<T extends InputFieldValue> = PayloadAction<{
  nodeId: string;
  fieldName: string;
  value: T['value'];
}>;

const fieldValueReducer = <T extends InputFieldValue>(
  state: NodesState,
  action: FieldValueAction<T>
) => {
  const { nodeId, fieldName, value } = action.payload;
  const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);
  const node = state.nodes?.[nodeIndex];
  if (!isInvocationNode(node)) {
    return;
  }
  const input = node.data?.inputs[fieldName];
  if (!input) {
    return;
  }
  if (nodeIndex > -1) {
    input.value = value;
  }
};

const nodesSlice = createSlice({
  name: 'nodes',
  initialState: initialNodesState,
  reducers: {
    nodesChanged: (state, action: PayloadAction<NodeChange[]>) => {
      state.nodes = applyNodeChanges(action.payload, state.nodes);
    },
    nodeAdded: (
      state,
      action: PayloadAction<
        Node<InvocationNodeData | CurrentImageNodeData | NotesNodeData>
      >
    ) => {
      const node = action.payload;
      const position = findUnoccupiedPosition(
        state.nodes,
        node.position.x,
        node.position.y
      );
      node.position = position;
      node.selected = true;

      state.nodes = applyNodeChanges(
        state.nodes.map((n) => ({ id: n.id, type: 'select', selected: false })),
        state.nodes
      );

      state.edges = applyEdgeChanges(
        state.edges.map((e) => ({ id: e.id, type: 'select', selected: false })),
        state.edges
      );

      state.nodes.push(node);

      if (!isInvocationNode(node)) {
        return;
      }

      state.nodeExecutionStates[node.id] = {
        nodeId: node.id,
        ...initialNodeExecutionState,
      };
    },
    edgesChanged: (state, action: PayloadAction<EdgeChange[]>) => {
      state.edges = applyEdgeChanges(action.payload, state.edges);
    },
    connectionStarted: (state, action: PayloadAction<OnConnectStartParams>) => {
      state.connectionStartParams = action.payload;
      const { nodeId, handleId, handleType } = action.payload;
      if (!nodeId || !handleId) {
        return;
      }
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);
      const node = state.nodes?.[nodeIndex];
      if (!isInvocationNode(node)) {
        return;
      }
      const field =
        handleType === 'source'
          ? node.data.outputs[handleId]
          : node.data.inputs[handleId];
      state.currentConnectionFieldType = field?.type ?? null;
    },
    connectionMade: (state, action: PayloadAction<Connection>) => {
      const fieldType = state.currentConnectionFieldType;
      if (!fieldType) {
        return;
      }
      state.edges = addEdge(
        { ...action.payload, type: 'default' },
        state.edges
      );
    },
    connectionEnded: (state) => {
      state.connectionStartParams = null;
      state.currentConnectionFieldType = null;
    },
    workflowExposedFieldAdded: (
      state,
      action: PayloadAction<FieldIdentifier>
    ) => {
      state.workflow.exposedFields = uniqBy(
        state.workflow.exposedFields.concat(action.payload),
        (field) => `${field.nodeId}-${field.fieldName}`
      );
    },
    workflowExposedFieldRemoved: (
      state,
      action: PayloadAction<FieldIdentifier>
    ) => {
      state.workflow.exposedFields = state.workflow.exposedFields.filter(
        (field) => !isEqual(field, action.payload)
      );
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
    nodeEmbedWorkflowChanged: (
      state,
      action: PayloadAction<{ nodeId: string; embedWorkflow: boolean }>
    ) => {
      const { nodeId, embedWorkflow } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);

      const node = state.nodes?.[nodeIndex];

      if (!isInvocationNode(node)) {
        return;
      }
      node.data.embedWorkflow = embedWorkflow;
    },
    nodeIsIntermediateChanged: (
      state,
      action: PayloadAction<{ nodeId: string; isIntermediate: boolean }>
    ) => {
      const { nodeId, isIntermediate } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);

      const node = state.nodes?.[nodeIndex];

      if (!isInvocationNode(node)) {
        return;
      }
      node.data.isIntermediate = isIntermediate;
    },
    nodeIsOpenChanged: (
      state,
      action: PayloadAction<{ nodeId: string; isOpen: boolean }>
    ) => {
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
        const closedIncomers = getIncomers(
          node,
          state.nodes,
          state.edges
        ).filter(
          (node) => isInvocationNode(node) && node.data.isOpen === false
        );

        const closedOutgoers = getOutgoers(
          node,
          state.nodes,
          state.edges
        ).filter(
          (node) => isInvocationNode(node) && node.data.isOpen === false
        );

        const collapsedEdgesToCreate: Edge<{ count: number }>[] = [];

        // hide all edges
        connectedEdges.forEach((edge) => {
          if (
            edge.target === nodeId &&
            closedIncomers.find((node) => node.id === edge.source)
          ) {
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
              });
            }
          }
          if (
            edge.source === nodeId &&
            closedOutgoers.find((node) => node.id === edge.target)
          ) {
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
              });
            }
          }
        });
        if (collapsedEdgesToCreate.length) {
          state.edges = applyEdgeChanges(
            collapsedEdgesToCreate.map((edge) => ({ type: 'add', item: edge })),
            state.edges
          );
        }
      }
    },
    edgesDeleted: (state, action: PayloadAction<Edge[]>) => {
      const edges = action.payload;
      const collapsedEdges = edges.filter((e) => e.type === 'collapsed');

      // if we delete a collapsed edge, we need to delete all collapsed edges between the same nodes
      if (collapsedEdges.length) {
        const edgeChanges: EdgeRemoveChange[] = [];
        collapsedEdges.forEach((collapsedEdge) => {
          state.edges.forEach((edge) => {
            if (
              edge.source === collapsedEdge.source &&
              edge.target === collapsedEdge.target
            ) {
              edgeChanges.push({ id: edge.id, type: 'remove' });
            }
          });
        });
        state.edges = applyEdgeChanges(edgeChanges, state.edges);
      }
    },
    nodesDeleted: (
      state,
      action: PayloadAction<
        Node<InvocationNodeData | NotesNodeData | CurrentImageNodeData>[]
      >
    ) => {
      action.payload.forEach((node) => {
        state.workflow.exposedFields = state.workflow.exposedFields.filter(
          (f) => f.nodeId !== node.id
        );
        if (!isInvocationNode(node)) {
          return;
        }
        delete state.nodeExecutionStates[node.id];
      });
    },
    nodeLabelChanged: (
      state,
      action: PayloadAction<{ nodeId: string; label: string }>
    ) => {
      const { nodeId, label } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);
      const node = state.nodes?.[nodeIndex];
      if (!isInvocationNode(node)) {
        return;
      }
      node.data.label = label;
    },
    nodeNotesChanged: (
      state,
      action: PayloadAction<{ nodeId: string; notes: string }>
    ) => {
      const { nodeId, notes } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);
      const node = state.nodes?.[nodeIndex];
      if (!isInvocationNode(node)) {
        return;
      }
      node.data.notes = notes;
    },
    nodeExclusivelySelected: (state, action: PayloadAction<string>) => {
      const nodeId = action.payload;
      state.nodes = applyNodeChanges(
        state.nodes.map((n) => ({
          id: n.id,
          type: 'select',
          selected: n.id === nodeId ? true : false,
        })),
        state.nodes
      );
    },
    selectedNodesChanged: (state, action: PayloadAction<string[]>) => {
      state.selectedNodes = action.payload;
    },
    selectedEdgesChanged: (state, action: PayloadAction<string[]>) => {
      state.selectedEdges = action.payload;
    },
    fieldStringValueChanged: (
      state,
      action: FieldValueAction<StringInputFieldValue>
    ) => {
      fieldValueReducer(state, action);
    },
    fieldNumberValueChanged: (
      state,
      action: FieldValueAction<IntegerInputFieldValue | FloatInputFieldValue>
    ) => {
      fieldValueReducer(state, action);
    },
    fieldBooleanValueChanged: (
      state,
      action: FieldValueAction<BooleanInputFieldValue>
    ) => {
      fieldValueReducer(state, action);
    },
    fieldImageValueChanged: (
      state,
      action: FieldValueAction<ImageInputFieldValue>
    ) => {
      fieldValueReducer(state, action);
    },
    fieldColorValueChanged: (
      state,
      action: FieldValueAction<ColorInputFieldValue>
    ) => {
      fieldValueReducer(state, action);
    },
    fieldMainModelValueChanged: (
      state,
      action: FieldValueAction<MainModelInputFieldValue>
    ) => {
      fieldValueReducer(state, action);
    },
    fieldRefinerModelValueChanged: (
      state,
      action: FieldValueAction<SDXLRefinerModelInputFieldValue>
    ) => {
      fieldValueReducer(state, action);
    },
    fieldVaeModelValueChanged: (
      state,
      action: FieldValueAction<VaeModelInputFieldValue>
    ) => {
      fieldValueReducer(state, action);
    },
    fieldLoRAModelValueChanged: (
      state,
      action: FieldValueAction<LoRAModelInputFieldValue>
    ) => {
      fieldValueReducer(state, action);
    },
    fieldControlNetModelValueChanged: (
      state,
      action: FieldValueAction<ControlNetModelInputFieldValue>
    ) => {
      fieldValueReducer(state, action);
    },
    fieldEnumModelValueChanged: (
      state,
      action: FieldValueAction<EnumInputFieldValue>
    ) => {
      fieldValueReducer(state, action);
    },
    fieldSchedulerValueChanged: (
      state,
      action: FieldValueAction<SchedulerInputFieldValue>
    ) => {
      fieldValueReducer(state, action);
    },
    imageCollectionFieldValueChanged: (
      state,
      action: PayloadAction<{
        nodeId: string;
        fieldName: string;
        value: ImageField[];
      }>
    ) => {
      const { nodeId, fieldName, value } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);

      if (nodeIndex === -1) {
        return;
      }

      const node = state.nodes?.[nodeIndex];

      if (!isInvocationNode(node)) {
        return;
      }

      const input = node.data?.inputs[fieldName];
      if (!input) {
        return;
      }

      const currentValue = cloneDeep(input.value);

      if (!currentValue) {
        input.value = value;
        return;
      }

      input.value = uniqBy(
        (currentValue as ImageField[]).concat(value),
        'image_name'
      );
    },
    notesNodeValueChanged: (
      state,
      action: PayloadAction<{ nodeId: string; value: string }>
    ) => {
      const { nodeId, value } = action.payload;
      const nodeIndex = state.nodes.findIndex((n) => n.id === nodeId);
      const node = state.nodes?.[nodeIndex];
      if (!isNotesNode(node)) {
        return;
      }
      node.data.notes = value;
    },
    shouldShowFieldTypeLegendChanged: (
      state,
      action: PayloadAction<boolean>
    ) => {
      state.shouldShowFieldTypeLegend = action.payload;
    },
    shouldShowMinimapPanelChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldShowMinimapPanel = action.payload;
    },
    nodeTemplatesBuilt: (
      state,
      action: PayloadAction<Record<string, InvocationTemplate>>
    ) => {
      state.nodeTemplates = action.payload;
      state.isReady = true;
    },
    nodeEditorReset: (state) => {
      state.nodes = [];
      state.edges = [];
      state.workflow = cloneDeep(initialWorkflow);
    },
    shouldValidateGraphChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldValidateGraph = action.payload;
    },
    shouldAnimateEdgesChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldAnimateEdges = action.payload;
    },
    shouldSnapToGridChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldSnapToGrid = action.payload;
    },
    shouldColorEdgesChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldColorEdges = action.payload;
    },
    nodeOpacityChanged: (state, action: PayloadAction<number>) => {
      state.nodeOpacity = action.payload;
    },
    workflowNameChanged: (state, action: PayloadAction<string>) => {
      state.workflow.name = action.payload;
    },
    workflowDescriptionChanged: (state, action: PayloadAction<string>) => {
      state.workflow.description = action.payload;
    },
    workflowTagsChanged: (state, action: PayloadAction<string>) => {
      state.workflow.tags = action.payload;
    },
    workflowAuthorChanged: (state, action: PayloadAction<string>) => {
      state.workflow.author = action.payload;
    },
    workflowNotesChanged: (state, action: PayloadAction<string>) => {
      state.workflow.notes = action.payload;
    },
    workflowVersionChanged: (state, action: PayloadAction<string>) => {
      state.workflow.version = action.payload;
    },
    workflowContactChanged: (state, action: PayloadAction<string>) => {
      state.workflow.contact = action.payload;
    },
    workflowLoaded: (state, action: PayloadAction<Workflow>) => {
      const { nodes, edges, ...workflow } = action.payload;
      state.workflow = workflow;

      state.nodes = applyNodeChanges(
        nodes.map((node) => ({
          item: { ...node, dragHandle: `.${DRAG_HANDLE_CLASSNAME}` },
          type: 'add',
        })),
        []
      );
      state.edges = applyEdgeChanges(
        edges.map((edge) => ({ item: edge, type: 'add' })),
        []
      );

      state.nodeExecutionStates = nodes.reduce<
        Record<string, NodeExecutionState>
      >((acc, node) => {
        acc[node.id] = {
          nodeId: node.id,
          ...initialNodeExecutionState,
        };
        return acc;
      }, {});
    },
    workflowReset: (state) => {
      state.workflow = cloneDeep(initialWorkflow);
    },
    viewportChanged: (state, action: PayloadAction<Viewport>) => {
      state.viewport = action.payload;
    },
    mouseOverFieldChanged: (
      state,
      action: PayloadAction<FieldIdentifier | null>
    ) => {
      state.mouseOverField = action.payload;
    },
    mouseOverNodeChanged: (state, action: PayloadAction<string | null>) => {
      state.mouseOverNode = action.payload;
    },
    selectedAll: (state) => {
      state.nodes = applyNodeChanges(
        state.nodes.map((n) => ({ id: n.id, type: 'select', selected: true })),
        state.nodes
      );
      state.edges = applyEdgeChanges(
        state.edges.map((e) => ({ id: e.id, type: 'select', selected: true })),
        state.edges
      );
    },
    selectionCopied: (state) => {
      state.nodesToCopy = state.nodes.filter((n) => n.selected).map(cloneDeep);
      state.edgesToCopy = state.edges.filter((e) => e.selected).map(cloneDeep);
    },
    selectionPasted: (state) => {
      const newNodes = state.nodesToCopy.map(cloneDeep);
      const oldNodeIds = newNodes.map((n) => n.data.id);
      const newEdges = state.edgesToCopy
        .filter(
          (e) => oldNodeIds.includes(e.source) && oldNodeIds.includes(e.target)
        )
        .map(cloneDeep);

      newEdges.forEach((e) => (e.selected = true));

      newNodes.forEach((node) => {
        const newNodeId = uuidv4();
        newEdges.forEach((edge) => {
          if (edge.source === node.data.id) {
            edge.source = newNodeId;
            edge.id = edge.id.replace(node.data.id, newNodeId);
          }
          if (edge.target === node.data.id) {
            edge.target = newNodeId;
            edge.id = edge.id.replace(node.data.id, newNodeId);
          }
        });
        node.selected = true;
        node.id = newNodeId;
        node.data.id = newNodeId;

        const position = findUnoccupiedPosition(
          state.nodes,
          node.position.x,
          node.position.y
        );

        node.position = position;
      });

      const nodeAdditions: NodeChange[] = newNodes.map((n) => ({
        item: n,
        type: 'add',
      }));
      const nodeSelectionChanges: NodeChange[] = state.nodes.map((n) => ({
        id: n.data.id,
        type: 'select',
        selected: false,
      }));

      const edgeAdditions: EdgeChange[] = newEdges.map((e) => ({
        item: e,
        type: 'add',
      }));
      const edgeSelectionChanges: EdgeChange[] = state.edges.map((e) => ({
        id: e.id,
        type: 'select',
        selected: false,
      }));

      state.nodes = applyNodeChanges(
        nodeAdditions.concat(nodeSelectionChanges),
        state.nodes
      );

      state.edges = applyEdgeChanges(
        edgeAdditions.concat(edgeSelectionChanges),
        state.edges
      );

      newNodes.forEach((node) => {
        state.nodeExecutionStates[node.id] = {
          nodeId: node.id,
          ...initialNodeExecutionState,
        };
      });
    },
    addNodePopoverOpened: (state) => {
      state.isAddNodePopoverOpen = true;
    },
    addNodePopoverClosed: (state) => {
      state.isAddNodePopoverOpen = false;
    },
    addNodePopoverToggled: (state) => {
      state.isAddNodePopoverOpen = !state.isAddNodePopoverOpen;
    },
    selectionModeChanged: (state, action: PayloadAction<boolean>) => {
      state.selectionMode = action.payload
        ? SelectionMode.Full
        : SelectionMode.Partial;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(receivedOpenAPISchema.pending, (state) => {
      state.isReady = false;
    });
    builder.addCase(appSocketInvocationStarted, (state, action) => {
      const { source_node_id } = action.payload.data;
      const node = state.nodeExecutionStates[source_node_id];
      if (node) {
        node.status = NodeStatus.IN_PROGRESS;
      }
    });
    builder.addCase(appSocketInvocationComplete, (state, action) => {
      const { source_node_id, result } = action.payload.data;
      const nes = state.nodeExecutionStates[source_node_id];
      if (nes) {
        nes.status = NodeStatus.COMPLETED;
        if (nes.progress !== null) {
          nes.progress = 1;
        }
        nes.outputs.push(result);
      }
    });
    builder.addCase(appSocketInvocationError, (state, action) => {
      const { source_node_id } = action.payload.data;
      const node = state.nodeExecutionStates[source_node_id];
      if (node) {
        node.status = NodeStatus.FAILED;
        node.error = action.payload.data.error;
        node.progress = null;
        node.progressImage = null;
      }
    });
    builder.addCase(appSocketGeneratorProgress, (state, action) => {
      const { source_node_id, step, total_steps, progress_image } =
        action.payload.data;
      const node = state.nodeExecutionStates[source_node_id];
      if (node) {
        node.status = NodeStatus.IN_PROGRESS;
        node.progress = (step + 1) / total_steps;
        node.progressImage = progress_image ?? null;
      }
    });
    builder.addCase(sessionInvoked.fulfilled, (state) => {
      forEach(state.nodeExecutionStates, (nes) => {
        nes.status = NodeStatus.PENDING;
        nes.error = null;
        nes.progress = null;
        nes.progressImage = null;
        nes.outputs = [];
      });
    });
    builder.addCase(sessionCanceled.fulfilled, (state) => {
      map(state.nodeExecutionStates, (nes) => {
        if (nes.status === NodeStatus.IN_PROGRESS) {
          nes.status = NodeStatus.PENDING;
        }
      });
    });
  },
});

export const {
  nodesChanged,
  edgesChanged,
  nodeAdded,
  nodesDeleted,
  connectionMade,
  connectionStarted,
  connectionEnded,
  shouldShowFieldTypeLegendChanged,
  shouldShowMinimapPanelChanged,
  nodeTemplatesBuilt,
  nodeEditorReset,
  imageCollectionFieldValueChanged,
  fieldStringValueChanged,
  fieldNumberValueChanged,
  fieldBooleanValueChanged,
  fieldImageValueChanged,
  fieldColorValueChanged,
  fieldMainModelValueChanged,
  fieldVaeModelValueChanged,
  fieldLoRAModelValueChanged,
  fieldEnumModelValueChanged,
  fieldControlNetModelValueChanged,
  fieldRefinerModelValueChanged,
  fieldSchedulerValueChanged,
  nodeIsOpenChanged,
  nodeLabelChanged,
  nodeNotesChanged,
  edgesDeleted,
  shouldValidateGraphChanged,
  shouldAnimateEdgesChanged,
  nodeOpacityChanged,
  shouldSnapToGridChanged,
  shouldColorEdgesChanged,
  selectedNodesChanged,
  selectedEdgesChanged,
  workflowNameChanged,
  workflowDescriptionChanged,
  workflowTagsChanged,
  workflowAuthorChanged,
  workflowNotesChanged,
  workflowVersionChanged,
  workflowContactChanged,
  workflowLoaded,
  notesNodeValueChanged,
  workflowExposedFieldAdded,
  workflowExposedFieldRemoved,
  fieldLabelChanged,
  viewportChanged,
  mouseOverFieldChanged,
  selectionCopied,
  selectionPasted,
  selectedAll,
  addNodePopoverOpened,
  addNodePopoverClosed,
  addNodePopoverToggled,
  selectionModeChanged,
  nodeEmbedWorkflowChanged,
  nodeIsIntermediateChanged,
  mouseOverNodeChanged,
  nodeExclusivelySelected,
} = nodesSlice.actions;

export default nodesSlice.reducer;
