import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { cloneDeep, forEach, isEqual, uniqBy } from 'lodash-es';
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
  updateEdge,
  Viewport,
  XYPosition,
} from 'reactflow';
import { receivedOpenAPISchema } from 'services/api/thunks/schema';
import { ImageField } from 'services/api/types';
import {
  appSocketGeneratorProgress,
  appSocketInvocationComplete,
  appSocketInvocationError,
  appSocketInvocationStarted,
  appSocketQueueItemStatusChanged,
} from 'services/events/actions';
import { v4 as uuidv4 } from 'uuid';
import { DRAG_HANDLE_CLASSNAME } from '../types/constants';
import {
  BoardInputFieldValue,
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
  IPAdapterModelInputFieldValue,
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
import { findConnectionToValidHandle } from './util/findConnectionToValidHandle';

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
  past: [],
  present: {
    nodes: [],
    edges: [],
    nodeTemplates: {},
    isReady: false,
    connectionStartParams: null,
    currentConnectionFieldType: null,
    connectionMade: false,
    modifyingEdge: false,
    addNewNodePosition: null,
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
  },
  future: [],
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
  const nodeIndex = state.present.nodes.findIndex((n) => n.id === nodeId);
  const node = state.present.nodes?.[nodeIndex];
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
    undo: (state) => {
      if (state.past.length > 0) {
        const previousState = state.past.pop()!;
        state.future.unshift(state.present);
        state.present = previousState;
      }
    },
    redo: (state) => {
      if (state.future.length > 0) {
        const nextState = state.future.shift()!;
        state.past.push(state.present);
        state.present = nextState;
      }
    },
    nodesChanged: (state, action: PayloadAction<NodeChange[]>) => {
      state.present.nodes = applyNodeChanges(
        action.payload,
        state.present.nodes
      );
    },
    nodeAdded: (
      state,
      action: PayloadAction<
        Node<InvocationNodeData | CurrentImageNodeData | NotesNodeData>
      >
    ) => {
      state.past.push(state.present);
      state.future = [];
      const node = action.payload;
      const position = findUnoccupiedPosition(
        state.present.nodes,
        state.present.addNewNodePosition?.x ?? node.position.x,
        state.present.addNewNodePosition?.y ?? node.position.y
      );
      node.position = position;
      node.selected = true;

      state.present.nodes = applyNodeChanges(
        state.present.nodes.map((n) => ({
          id: n.id,
          type: 'select',
          selected: false,
        })),
        state.present.nodes
      );

      state.present.edges = applyEdgeChanges(
        state.present.edges.map((e) => ({
          id: e.id,
          type: 'select',
          selected: false,
        })),
        state.present.edges
      );

      state.present.nodes.push(node);

      if (!isInvocationNode(node)) {
        return;
      }

      state.present.nodeExecutionStates[node.id] = {
        nodeId: node.id,
        ...initialNodeExecutionState,
      };

      if (state.present.connectionStartParams) {
        const { nodeId, handleId, handleType } =
          state.present.connectionStartParams;
        if (
          nodeId &&
          handleId &&
          handleType &&
          state.present.currentConnectionFieldType
        ) {
          const newConnection = findConnectionToValidHandle(
            node,
            state.present.nodes,
            state.present.edges,
            nodeId,
            handleId,
            handleType,
            state.present.currentConnectionFieldType
          );
          if (newConnection) {
            state.present.edges = addEdge(
              { ...newConnection, type: 'default' },
              state.present.edges
            );
          }
        }
      }

      state.present.connectionStartParams = null;
      state.present.currentConnectionFieldType = null;
    },
    edgeChangeStarted: (state) => {
      state.present.modifyingEdge = true;
    },
    edgesChanged: (state, action: PayloadAction<EdgeChange[]>) => {
      state.present.edges = applyEdgeChanges(
        action.payload,
        state.present.edges
      );
    },
    edgeAdded: (state, action: PayloadAction<Edge>) => {
      state.past.push(state.present);
      state.future = [];
      state.present.edges = addEdge(action.payload, state.present.edges);
    },
    edgeUpdated: (
      state,
      action: PayloadAction<{ oldEdge: Edge; newConnection: Connection }>
    ) => {
      const { oldEdge, newConnection } = action.payload;
      state.present.edges = updateEdge(
        oldEdge,
        newConnection,
        state.present.edges
      );
    },
    connectionStarted: (state, action: PayloadAction<OnConnectStartParams>) => {
      state.present.connectionStartParams = action.payload;
      state.present.connectionMade = state.present.modifyingEdge;
      const { nodeId, handleId, handleType } = action.payload;
      if (!nodeId || !handleId) {
        return;
      }
      const nodeIndex = state.present.nodes.findIndex((n) => n.id === nodeId);
      const node = state.present.nodes?.[nodeIndex];
      if (!isInvocationNode(node)) {
        return;
      }
      const field =
        handleType === 'source'
          ? node.data.outputs[handleId]
          : node.data.inputs[handleId];
      state.present.currentConnectionFieldType = field?.type ?? null;
    },
    connectionMade: (state, action: PayloadAction<Connection>) => {
      const fieldType = state.present.currentConnectionFieldType;
      if (!fieldType) {
        return;
      }
      state.present.edges = addEdge(
        { ...action.payload, type: 'default' },
        state.present.edges
      );

      state.present.connectionMade = true;
    },
    connectionEnded: (state, action) => {
      if (!state.present.connectionMade) {
        if (state.present.mouseOverNode) {
          const nodeIndex = state.present.nodes.findIndex(
            (n) => n.id === state.present.mouseOverNode
          );
          const mouseOverNode = state.present.nodes?.[nodeIndex];
          if (mouseOverNode && state.present.connectionStartParams) {
            const { nodeId, handleId, handleType } =
              state.present.connectionStartParams;
            if (
              nodeId &&
              handleId &&
              handleType &&
              state.present.currentConnectionFieldType
            ) {
              const newConnection = findConnectionToValidHandle(
                mouseOverNode,
                state.present.nodes,
                state.present.edges,
                nodeId,
                handleId,
                handleType,
                state.present.currentConnectionFieldType
              );
              if (newConnection) {
                state.present.edges = addEdge(
                  { ...newConnection, type: 'default' },
                  state.present.edges
                );
              }
            }
          }
          state.present.connectionStartParams = null;
          state.present.currentConnectionFieldType = null;
        } else {
          state.present.addNewNodePosition = action.payload.cursorPosition;
          state.present.isAddNodePopoverOpen = true;
        }
      } else {
        state.present.connectionStartParams = null;
        state.present.currentConnectionFieldType = null;
      }
      state.present.modifyingEdge = false;
    },
    workflowExposedFieldAdded: (
      state,
      action: PayloadAction<FieldIdentifier>
    ) => {
      state.present.workflow.exposedFields = uniqBy(
        state.present.workflow.exposedFields.concat(action.payload),
        (field) => `${field.nodeId}-${field.fieldName}`
      );
    },
    workflowExposedFieldRemoved: (
      state,
      action: PayloadAction<FieldIdentifier>
    ) => {
      state.present.workflow.exposedFields =
        state.present.workflow.exposedFields.filter(
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
      const node = state.present.nodes.find((n) => n.id === nodeId);
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
      const nodeIndex = state.present.nodes.findIndex((n) => n.id === nodeId);

      const node = state.present.nodes?.[nodeIndex];

      if (!isInvocationNode(node)) {
        return;
      }
      node.data.embedWorkflow = embedWorkflow;
    },
    nodeUseCacheChanged: (
      state,
      action: PayloadAction<{ nodeId: string; useCache: boolean }>
    ) => {
      const { nodeId, useCache } = action.payload;
      const nodeIndex = state.present.nodes.findIndex((n) => n.id === nodeId);

      const node = state.present.nodes?.[nodeIndex];

      if (!isInvocationNode(node)) {
        return;
      }
      node.data.useCache = useCache;
    },
    nodeIsIntermediateChanged: (
      state,
      action: PayloadAction<{ nodeId: string; isIntermediate: boolean }>
    ) => {
      const { nodeId, isIntermediate } = action.payload;
      const nodeIndex = state.present.nodes.findIndex((n) => n.id === nodeId);

      const node = state.present.nodes?.[nodeIndex];

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
      const nodeIndex = state.present.nodes.findIndex((n) => n.id === nodeId);

      const node = state.present.nodes?.[nodeIndex];
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

      const connectedEdges = getConnectedEdges([node], state.present.edges);

      if (isOpen) {
        // reset hidden status of all edges
        connectedEdges.forEach((edge) => {
          delete edge.hidden;
        });
        // delete dummy edges
        connectedEdges.forEach((edge) => {
          if (edge.type === 'collapsed') {
            state.present.edges = state.present.edges.filter(
              (e) => e.id !== edge.id
            );
          }
        });
      } else {
        const closedIncomers = getIncomers(
          node,
          state.present.nodes,
          state.present.edges
        ).filter(
          (node) => isInvocationNode(node) && node.data.isOpen === false
        );

        const closedOutgoers = getOutgoers(
          node,
          state.present.nodes,
          state.present.edges
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
                updatable: false,
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
                updatable: false,
              });
            }
          }
        });
        if (collapsedEdgesToCreate.length) {
          state.present.edges = applyEdgeChanges(
            collapsedEdgesToCreate.map((edge) => ({ type: 'add', item: edge })),
            state.present.edges
          );
        }
      }
    },
    edgeDeleted: (state, action: PayloadAction<string>) => {
      state.past.push(state.present);
      state.future = [];
      state.present.edges = state.present.edges.filter(
        (e) => e.id !== action.payload
      );
    },
    edgesDeleted: (state, action: PayloadAction<Edge[]>) => {
      state.past.push(state.present);
      state.future = [];
      const edges = action.payload;
      const collapsedEdges = edges.filter((e) => e.type === 'collapsed');

      // if we delete a collapsed edge, we need to delete all collapsed edges between the same nodes
      if (collapsedEdges.length) {
        const edgeChanges: EdgeRemoveChange[] = [];
        collapsedEdges.forEach((collapsedEdge) => {
          state.present.edges.forEach((edge) => {
            if (
              edge.source === collapsedEdge.source &&
              edge.target === collapsedEdge.target
            ) {
              edgeChanges.push({ id: edge.id, type: 'remove' });
            }
          });
        });
        state.present.edges = applyEdgeChanges(
          edgeChanges,
          state.present.edges
        );
      }
    },
    nodesDeleted: (
      state,
      action: PayloadAction<
        Node<InvocationNodeData | NotesNodeData | CurrentImageNodeData>[]
      >
    ) => {
      state.past.push(state.present);
      state.future = [];

      action.payload.forEach((node) => {
        state.present.workflow.exposedFields =
          state.present.workflow.exposedFields.filter(
            (f) => f.nodeId !== node.id
          );
        if (!isInvocationNode(node)) {
          return;
        }
        delete state.present.nodeExecutionStates[node.id];
      });
    },
    nodeLabelChanged: (
      state,
      action: PayloadAction<{ nodeId: string; label: string }>
    ) => {
      const { nodeId, label } = action.payload;
      const nodeIndex = state.present.nodes.findIndex((n) => n.id === nodeId);
      const node = state.present.nodes?.[nodeIndex];
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
      const nodeIndex = state.present.nodes.findIndex((n) => n.id === nodeId);
      const node = state.present.nodes?.[nodeIndex];
      if (!isInvocationNode(node)) {
        return;
      }
      node.data.notes = notes;
    },
    nodeExclusivelySelected: (state, action: PayloadAction<string>) => {
      const nodeId = action.payload;
      state.present.nodes = applyNodeChanges(
        state.present.nodes.map((n) => ({
          id: n.id,
          type: 'select',
          selected: n.id === nodeId ? true : false,
        })),
        state.present.nodes
      );
    },
    selectedNodesChanged: (state, action: PayloadAction<string[]>) => {
      state.present.selectedNodes = action.payload;
    },
    selectedEdgesChanged: (state, action: PayloadAction<string[]>) => {
      state.present.selectedEdges = action.payload;
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
    fieldBoardValueChanged: (
      state,
      action: FieldValueAction<BoardInputFieldValue>
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
    fieldIPAdapterModelValueChanged: (
      state,
      action: FieldValueAction<IPAdapterModelInputFieldValue>
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
      const nodeIndex = state.present.nodes.findIndex((n) => n.id === nodeId);

      if (nodeIndex === -1) {
        return;
      }

      const node = state.present.nodes?.[nodeIndex];

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
      const nodeIndex = state.present.nodes.findIndex((n) => n.id === nodeId);
      const node = state.present.nodes?.[nodeIndex];
      if (!isNotesNode(node)) {
        return;
      }
      node.data.notes = value;
    },
    shouldShowFieldTypeLegendChanged: (
      state,
      action: PayloadAction<boolean>
    ) => {
      state.present.shouldShowFieldTypeLegend = action.payload;
    },
    shouldShowMinimapPanelChanged: (state, action: PayloadAction<boolean>) => {
      state.present.shouldShowMinimapPanel = action.payload;
    },
    nodeTemplatesBuilt: (
      state,
      action: PayloadAction<Record<string, InvocationTemplate>>
    ) => {
      state.present.nodeTemplates = action.payload;
      state.present.isReady = true;
    },
    nodeEditorReset: (state) => {
      state.present.nodes = [];
      state.present.edges = [];
      state.present.workflow = cloneDeep(initialWorkflow);
    },
    shouldValidateGraphChanged: (state, action: PayloadAction<boolean>) => {
      state.present.shouldValidateGraph = action.payload;
    },
    shouldAnimateEdgesChanged: (state, action: PayloadAction<boolean>) => {
      state.present.shouldAnimateEdges = action.payload;
    },
    shouldSnapToGridChanged: (state, action: PayloadAction<boolean>) => {
      state.present.shouldSnapToGrid = action.payload;
    },
    shouldColorEdgesChanged: (state, action: PayloadAction<boolean>) => {
      state.present.shouldColorEdges = action.payload;
    },
    nodeOpacityChanged: (state, action: PayloadAction<number>) => {
      state.present.nodeOpacity = action.payload;
    },
    workflowNameChanged: (state, action: PayloadAction<string>) => {
      state.present.workflow.name = action.payload;
    },
    workflowDescriptionChanged: (state, action: PayloadAction<string>) => {
      state.present.workflow.description = action.payload;
    },
    workflowTagsChanged: (state, action: PayloadAction<string>) => {
      state.present.workflow.tags = action.payload;
    },
    workflowAuthorChanged: (state, action: PayloadAction<string>) => {
      state.present.workflow.author = action.payload;
    },
    workflowNotesChanged: (state, action: PayloadAction<string>) => {
      state.present.workflow.notes = action.payload;
    },
    workflowVersionChanged: (state, action: PayloadAction<string>) => {
      state.present.workflow.version = action.payload;
    },
    workflowContactChanged: (state, action: PayloadAction<string>) => {
      state.present.workflow.contact = action.payload;
    },
    workflowLoaded: (state, action: PayloadAction<Workflow>) => {
      const { nodes, edges, ...workflow } = action.payload;
      state.present.workflow = workflow;

      state.present.nodes = applyNodeChanges(
        nodes.map((node) => ({
          item: { ...node, dragHandle: `.${DRAG_HANDLE_CLASSNAME}` },
          type: 'add',
        })),
        []
      );
      state.present.edges = applyEdgeChanges(
        edges.map((edge) => ({ item: edge, type: 'add' })),
        []
      );

      state.present.nodeExecutionStates = nodes.reduce<
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
      state.present.workflow = cloneDeep(initialWorkflow);
    },
    viewportChanged: (state, action: PayloadAction<Viewport>) => {
      state.present.viewport = action.payload;
    },
    mouseOverFieldChanged: (
      state,
      action: PayloadAction<FieldIdentifier | null>
    ) => {
      state.present.mouseOverField = action.payload;
    },
    mouseOverNodeChanged: (state, action: PayloadAction<string | null>) => {
      state.present.mouseOverNode = action.payload;
    },
    selectedAll: (state) => {
      state.present.nodes = applyNodeChanges(
        state.present.nodes.map((n) => ({
          id: n.id,
          type: 'select',
          selected: true,
        })),
        state.present.nodes
      );
      state.present.edges = applyEdgeChanges(
        state.present.edges.map((e) => ({
          id: e.id,
          type: 'select',
          selected: true,
        })),
        state.present.edges
      );
    },
    selectionCopied: (state) => {
      state.present.nodesToCopy = state.present.nodes
        .filter((n) => n.selected)
        .map(cloneDeep);
      state.present.edgesToCopy = state.present.edges
        .filter((e) => e.selected)
        .map(cloneDeep);

      if (state.present.nodesToCopy.length > 0) {
        const averagePosition = { x: 0, y: 0 };
        state.present.nodesToCopy.forEach((e) => {
          const xOffset = 0.15 * (e.width ?? 0);
          const yOffset = 0.5 * (e.height ?? 0);
          averagePosition.x += e.position.x + xOffset;
          averagePosition.y += e.position.y + yOffset;
        });

        averagePosition.x /= state.present.nodesToCopy.length;
        averagePosition.y /= state.present.nodesToCopy.length;

        state.present.nodesToCopy.forEach((e) => {
          e.position.x -= averagePosition.x;
          e.position.y -= averagePosition.y;
        });
      }
    },
    selectionPasted: (
      state,
      action: PayloadAction<{ cursorPosition?: XYPosition }>
    ) => {
      const { cursorPosition } = action.payload;
      const newNodes = state.present.nodesToCopy.map(cloneDeep);
      const oldNodeIds = newNodes.map((n) => n.data.id);
      const newEdges = state.present.edgesToCopy
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
          state.present.nodes,
          node.position.x + (cursorPosition?.x ?? 0),
          node.position.y + (cursorPosition?.y ?? 0)
        );

        node.position = position;
      });

      const nodeAdditions: NodeChange[] = newNodes.map((n) => ({
        item: n,
        type: 'add',
      }));
      const nodeSelectionChanges: NodeChange[] = state.present.nodes.map(
        (n) => ({
          id: n.data.id,
          type: 'select',
          selected: false,
        })
      );

      const edgeAdditions: EdgeChange[] = newEdges.map((e) => ({
        item: e,
        type: 'add',
      }));
      const edgeSelectionChanges: EdgeChange[] = state.present.edges.map(
        (e) => ({
          id: e.id,
          type: 'select',
          selected: false,
        })
      );

      state.present.nodes = applyNodeChanges(
        nodeAdditions.concat(nodeSelectionChanges),
        state.present.nodes
      );

      state.present.edges = applyEdgeChanges(
        edgeAdditions.concat(edgeSelectionChanges),
        state.present.edges
      );

      newNodes.forEach((node) => {
        state.present.nodeExecutionStates[node.id] = {
          nodeId: node.id,
          ...initialNodeExecutionState,
        };
      });
    },
    addNodePopoverOpened: (state) => {
      state.present.addNewNodePosition = null; //Create the node in viewport center by default
      state.present.isAddNodePopoverOpen = true;
    },
    addNodePopoverClosed: (state) => {
      state.present.isAddNodePopoverOpen = false;

      //Make sure these get reset if we close the popover and haven't selected a node
      state.present.connectionStartParams = null;
      state.present.currentConnectionFieldType = null;
    },
    addNodePopoverToggled: (state) => {
      state.present.isAddNodePopoverOpen = !state.present.isAddNodePopoverOpen;
    },
    selectionModeChanged: (state, action: PayloadAction<boolean>) => {
      state.present.selectionMode = action.payload
        ? SelectionMode.Full
        : SelectionMode.Partial;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(receivedOpenAPISchema.pending, (state) => {
      state.present.isReady = false;
    });
    builder.addCase(appSocketInvocationStarted, (state, action) => {
      const { source_node_id } = action.payload.data;
      const node = state.present.nodeExecutionStates[source_node_id];
      if (node) {
        node.status = NodeStatus.IN_PROGRESS;
      }
    });
    builder.addCase(appSocketInvocationComplete, (state, action) => {
      const { source_node_id, result } = action.payload.data;
      const nes = state.present.nodeExecutionStates[source_node_id];
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
      const node = state.present.nodeExecutionStates[source_node_id];
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
      const node = state.present.nodeExecutionStates[source_node_id];
      if (node) {
        node.status = NodeStatus.IN_PROGRESS;
        node.progress = (step + 1) / total_steps;
        node.progressImage = progress_image ?? null;
      }
    });
    builder.addCase(appSocketQueueItemStatusChanged, (state, action) => {
      if (['in_progress'].includes(action.payload.data.status)) {
        forEach(state.present.nodeExecutionStates, (nes) => {
          nes.status = NodeStatus.PENDING;
          nes.error = null;
          nes.progress = null;
          nes.progressImage = null;
          nes.outputs = [];
        });
      }
    });
  },
});

export const {
  addNodePopoverClosed,
  addNodePopoverOpened,
  addNodePopoverToggled,
  connectionEnded,
  connectionMade,
  connectionStarted,
  edgeDeleted,
  edgeChangeStarted,
  edgesChanged,
  edgesDeleted,
  edgeUpdated,
  fieldBoardValueChanged,
  fieldBooleanValueChanged,
  fieldColorValueChanged,
  fieldControlNetModelValueChanged,
  fieldEnumModelValueChanged,
  fieldImageValueChanged,
  fieldIPAdapterModelValueChanged,
  fieldLabelChanged,
  fieldLoRAModelValueChanged,
  fieldMainModelValueChanged,
  fieldNumberValueChanged,
  fieldRefinerModelValueChanged,
  fieldSchedulerValueChanged,
  fieldStringValueChanged,
  fieldVaeModelValueChanged,
  imageCollectionFieldValueChanged,
  mouseOverFieldChanged,
  mouseOverNodeChanged,
  nodeAdded,
  nodeEditorReset,
  nodeEmbedWorkflowChanged,
  nodeExclusivelySelected,
  nodeIsIntermediateChanged,
  nodeIsOpenChanged,
  nodeLabelChanged,
  nodeNotesChanged,
  nodeOpacityChanged,
  nodesChanged,
  nodesDeleted,
  nodeTemplatesBuilt,
  nodeUseCacheChanged,
  notesNodeValueChanged,
  selectedAll,
  selectedEdgesChanged,
  selectedNodesChanged,
  selectionCopied,
  selectionModeChanged,
  selectionPasted,
  shouldAnimateEdgesChanged,
  shouldColorEdgesChanged,
  shouldShowFieldTypeLegendChanged,
  shouldShowMinimapPanelChanged,
  shouldSnapToGridChanged,
  shouldValidateGraphChanged,
  viewportChanged,
  workflowAuthorChanged,
  workflowContactChanged,
  workflowDescriptionChanged,
  workflowExposedFieldAdded,
  workflowExposedFieldRemoved,
  workflowLoaded,
  workflowNameChanged,
  workflowNotesChanged,
  workflowTagsChanged,
  workflowVersionChanged,
  edgeAdded,
  undo,
  redo,
} = nodesSlice.actions;

export default nodesSlice.reducer;
