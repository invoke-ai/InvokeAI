import type { ProjectGraphState, XYPosition } from '@features/workflow/contracts';
import type { WorkflowPerfSource, WorkflowRuntimeApi } from '@features/workflow/ui/contracts';

import { Box, Flex, HStack, Spinner, Stack, Text } from '@chakra-ui/react';
import { getProjectGraphReadiness } from '@features/workflow/graph';
import '@xyflow/react/dist/style.css';
import { ensureInvocationTemplatesLoaded, useInvocationTemplatesSelector } from '@features/workflow/react';
import { FlowMiniMap, flowThemeCss, getFlowColorMode } from '@features/workflow/ui/graph-preview';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import {
  useWorkflowNotifications,
  useWorkflowPreferencesSelector,
  useWorkflowProjectSelector,
  useWorkflowUi,
} from '@features/workflow/ui/WorkflowUiContext';
import { setAddNodeOpen } from '@features/workflow/ui/workflowUiStore';
import {
  buildConnectorNode,
  createWorkflowGraphIndex,
  createWorkflowId,
  getWorkflowSourceFieldType,
  getWorkflowTargetFieldType,
  validateConnection,
} from '@features/workflow/utility';
import { useModifierHeld } from '@platform/react/useModifierHeld';
import { shallowEqual } from '@platform/state/selectors';
import {
  Background,
  BackgroundVariant,
  ConnectionLineType,
  ReactFlow,
  ReactFlowProvider,
  SelectionMode,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type EdgeChange,
  type EdgeTypes,
  type IsValidConnection,
  type NodeChange,
  type NodeTypes,
  type OnConnectEnd,
  type Viewport,
} from '@xyflow/react';
import {
  useCallback,
  useEffect,
  useEffectEvent,
  useId,
  useMemo,
  useRef,
  useState,
  startTransition,
  type MouseEvent as ReactMouseEvent,
} from 'react';

import { buildDuplicateElements, buildPasteElements, copyNodesToClipboard, useHasClipboardNodes } from './clipboard';
import { ConnectorFlowNode } from './ConnectorFlowNode';
import { CurrentImageFlowNode } from './CurrentImageFlowNode';
import { EditorToolbar, type EditorTool } from './EditorToolbar';
import {
  toFlowEdges,
  toFlowNodes,
  withNodeSelection,
  type FlowEdgeType,
  type WorkflowFlowEdge,
  type WorkflowFlowNode,
} from './flowAdapters';
import {
  registerWorkflowFlowInstance,
  releaseWorkflowFlowInstance,
  type WorkflowFlowInstance,
} from './flowInstanceStore';
import { InvocationFlowNode } from './InvocationFlowNode';
import { NodeContextMenu, type WorkflowContextMenuState } from './NodeContextMenu';
import { NotesFlowNode } from './NotesFlowNode';
import {
  isLargeWorkflowGraph,
  WORKFLOW_INITIAL_RENDER_NODE_COUNT,
  WORKFLOW_MINIMAP_DELAY_MS,
} from './performanceConstants';
import {
  clearNodeSelectionRequest,
  reportNodeHover,
  reportNodeSelection,
  workflowSelectionStore,
} from './selectionStore';
import { useEraser } from './useEraser';
import { useLasso } from './useLasso';
import { WorkflowEdge } from './WorkflowEdge';
import { getWorkflowViewport, getWorkflowViewportKey, setWorkflowViewport } from './workflowViewportStore';

const nodeTypes: NodeTypes = {
  connector: ConnectorFlowNode,
  current_image: CurrentImageFlowNode,
  invocation: InvocationFlowNode,
  notes: NotesFlowNode,
};

const edgeTypes: EdgeTypes = {
  default: WorkflowEdge,
  step: WorkflowEdge,
};

/**
 * The workflow center view: an xyflow editor over the project graph document.
 * The document is the source of truth — flow state is rebuilt from it on every
 * document change (undo, import, field edits), while transient view state
 * (selection, in-flight drags, the active tool) lives in local component state.
 */
/** Snap spacing matches the background dot grid. */
const SNAP_GRID: [number, number] = [24, 24];

const DELETE_KEY_CODES = ['Backspace', 'Delete'];

const DEFAULT_EDGE_OPTIONS = { style: { strokeWidth: 2 } };
const DEFAULT_VIEWPORT = { x: 0, y: 0, zoom: 1 } as const;

interface WorkflowFlowModel {
  edges: WorkflowFlowEdge[];
  nodes: WorkflowFlowNode[];
}

const EMPTY_FLOW_EDGES: WorkflowFlowEdge[] = [];
const EMPTY_FLOW_NODES: WorkflowFlowNode[] = [];

const buildWorkflowFlowModel = ({
  document,
  edgeType,
  invocationTemplates,
  previousEdges = [],
  previousNodes = [],
  reduceMotion,
  selectedNodeIds = new Set<string>(),
  isCompact = false,
  canUseCache = false,
  perfSource,
  time,
}: {
  canUseCache?: boolean;
  document: ProjectGraphState;
  edgeType: FlowEdgeType;
  isCompact?: boolean;
  invocationTemplates?: Parameters<typeof toFlowNodes>[2];
  perfSource: WorkflowPerfSource;
  previousEdges?: WorkflowFlowEdge[];
  previousNodes?: WorkflowFlowNode[];
  reduceMotion: boolean;
  selectedNodeIds?: Set<string>;
  time: <T>(name: string, source: WorkflowPerfSource, callback: () => T) => T;
}): WorkflowFlowModel => {
  const index = time('workflow:create-graph-index', perfSource, () =>
    createWorkflowGraphIndex(document.nodes, document.edges)
  );

  return time('workflow:build-flow-model', perfSource, () => ({
    edges: time('workflow:to-flow-edges', perfSource, () =>
      toFlowEdges(document, previousEdges, edgeType, selectedNodeIds, invocationTemplates, reduceMotion, index)
    ),
    nodes: time('workflow:to-flow-nodes', perfSource, () =>
      toFlowNodes(document, previousNodes, invocationTemplates, index, isCompact, canUseCache)
    ),
  }));
};

const getEventClientPosition = (event: MouseEvent | TouchEvent): { x: number; y: number } | null => {
  if (event instanceof MouseEvent) {
    return { x: event.clientX, y: event.clientY };
  }

  const touch = event.changedTouches[0];

  return touch ? { x: touch.clientX, y: touch.clientY } : null;
};

const WorkflowEditorPreparingState = ({ edgeCount, nodeCount }: { edgeCount: number; nodeCount: number }) => (
  <Flex align="center" bg="bg.inset" h="full" justify="center" p="6" w="full">
    <Stack align="center" gap="3" textAlign="center">
      <HStack color="fg.muted" gap="2">
        <Spinner size="sm" />
        <Text fontSize="sm" fontWeight="700">
          Preparing workflow graph
        </Text>
      </HStack>
      <Text color="fg.subtle" fontSize="xs">
        Loading {nodeCount.toLocaleString()} node{nodeCount === 1 ? '' : 's'} and {edgeCount.toLocaleString()} edge
        {edgeCount === 1 ? '' : 's'}.
      </Text>
    </Stack>
  </Flex>
);

const getSelectedNodeIdSet = (nodes: WorkflowFlowNode[]): Set<string> =>
  new Set(nodes.filter((node) => node.selected).map((node) => node.id));

const getNodeDistanceFromViewportOrigin = (node: WorkflowFlowNode, viewport: Viewport): number => {
  const zoom = viewport.zoom || 1;
  const originX = -viewport.x / zoom;
  const originY = -viewport.y / zoom;
  const dx = node.position.x - originX;
  const dy = node.position.y - originY;

  return dx * dx + dy * dy;
};

export const getInitialRenderFlowModel = (model: WorkflowFlowModel, viewport: Viewport): WorkflowFlowModel => {
  if (model.nodes.length <= WORKFLOW_INITIAL_RENDER_NODE_COUNT) {
    return model;
  }

  const nodes = [...model.nodes]
    .sort(
      (left, right) =>
        getNodeDistanceFromViewportOrigin(left, viewport) - getNodeDistanceFromViewportOrigin(right, viewport)
    )
    .slice(0, WORKFLOW_INITIAL_RENDER_NODE_COUNT);
  const nodeIds = new Set(nodes.map((node) => node.id));
  const edges = model.edges.filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target));

  return { edges, nodes };
};

const WorkflowFlow = ({ runtime }: { runtime: WorkflowRuntimeApi }) => {
  const projectGraph = useWorkflowProjectSelector((project) => project.projectGraph);
  const projectId = useWorkflowProjectSelector((project) => project.id);
  const ui = useWorkflowUi();
  const { mark: markWorkbenchPerf, measure: measureWorkbenchPerf, time: timeWorkbenchPerf } = ui.performance;
  const { editGraph, redo, undo } = useProjectGraphCommands();
  const notify = useWorkflowNotifications();
  const {
    reduceMotion,
    themeId,
    workflowEdgeStyle,
    workflowShowMinimap,
    workflowSnapToGrid,
    workflowValidateConnections,
  } = useWorkflowPreferencesSelector(
    (preferences) => ({
      reduceMotion: preferences.reduceMotion,
      themeId: preferences.themeId,
      workflowEdgeStyle: preferences.workflowEdgeStyle,
      workflowShowMinimap: preferences.workflowShowMinimap,
      workflowSnapToGrid: preferences.workflowSnapToGrid,
      workflowValidateConnections: preferences.workflowValidateConnections,
    }),
    shallowEqual
  );
  const templatesStatus = useInvocationTemplatesSelector((snapshot) => snapshot.status);
  const templates = useInvocationTemplatesSelector((snapshot) => snapshot.templates);
  const invocationTemplates = templatesStatus === 'loaded' ? templates : undefined;
  const canUseCache = ui.canUseCache;
  const edgeType: FlowEdgeType = workflowEdgeStyle === 'square' ? 'step' : 'default';
  const isLargeGraph = isLargeWorkflowGraph({
    edgeCount: projectGraph.edges.length,
    nodeCount: projectGraph.nodes.length,
  });
  const viewportKey = useMemo(
    () => getWorkflowViewportKey(projectId, runtime.instanceId),
    [projectId, runtime.instanceId]
  );
  const perfSource = useMemo<WorkflowPerfSource>(
    () => ({
      instanceId: runtime.instanceId,
      kind: 'widget',
      projectId,
      region: runtime.region,
      typeId: runtime.typeId,
    }),
    [projectId, runtime.instanceId, runtime.region, runtime.typeId]
  );
  const defaultViewport = useMemo(() => getWorkflowViewport(viewportKey) ?? DEFAULT_VIEWPORT, [viewportKey]);
  const [flowModel, setFlowModel] = useState<WorkflowFlowModel | null>(() =>
    isLargeGraph
      ? null
      : buildWorkflowFlowModel({
          canUseCache,
          document: projectGraph,
          edgeType,
          invocationTemplates,
          isCompact: isLargeGraph,
          perfSource,
          reduceMotion,
          time: timeWorkbenchPerf,
        })
  );
  const flowNodes = flowModel?.nodes ?? EMPTY_FLOW_NODES;
  const flowEdges = flowModel?.edges ?? EMPTY_FLOW_EDGES;
  const isPreparing = flowModel === null;
  const shouldDeferInitialBuild = isLargeGraph && flowModel === null;
  const [flowInstance, setFlowInstance] = useState<WorkflowFlowInstance | null>(null);
  const [isFullGraphMounted, setIsFullGraphMounted] = useState(!isLargeGraph);
  const [tool, setTool] = useState<EditorTool>('pan');
  const [nodeOpacity, setNodeOpacity] = useState(1);
  const [isMinimapReady, setIsMinimapReady] = useState(!isLargeGraph);
  const [contextMenu, setContextMenu] = useState<WorkflowContextMenuState | null>(null);
  const selectedNodeIds = workflowSelectionStore.useSelector((snapshot) => snapshot.selectedNodeIds);
  const isSnapHeld = useModifierHeld('Control');
  const hasClipboardNodes = useHasClipboardNodes();
  const backgroundId = useId().replace(/:/g, '');
  const perfMountMarkRef = useRef(`workflow:${runtime.instanceId}:editor-mounted`);
  const perfReadyMarkRef = useRef(`workflow:${runtime.instanceId}:editor-ready`);
  const hasScheduledFullGraphMountRef = useRef(false);
  /** Node ids to select once the next document-driven rebuild lands (fresh paste/duplicate results). */
  const pendingSelectionRef = useRef<string[] | null>(null);
  const lastBuiltModelRef = useRef<{
    edgeType: FlowEdgeType;
    invocationTemplates: typeof invocationTemplates;
    projectGraph: ProjectGraphState;
    reduceMotion: boolean;
    selectedNodeIds: string[];
  } | null>(null);

  const selectNodes = useCallback(
    (nodeIds: string[]) => {
      const selectedNodeIdSet = new Set(nodeIds);

      setFlowModel((current) => {
        if (!current) {
          return current;
        }

        const index = createWorkflowGraphIndex(projectGraph.nodes, projectGraph.edges);

        return {
          edges: toFlowEdges(
            projectGraph,
            current.edges,
            edgeType,
            selectedNodeIdSet,
            invocationTemplates,
            reduceMotion,
            index
          ),
          nodes: withNodeSelection(current.nodes, selectedNodeIdSet),
        };
      });
      reportNodeSelection(nodeIds);
    },
    [edgeType, invocationTemplates, projectGraph, reduceMotion]
  );

  const { lassoHandlers, lassoOverlay } = useLasso({
    enabled: tool === 'lasso',
    flowInstance,
    onSelect: selectNodes,
  });
  const eraseElements = useCallback(
    ({ edgeIds, nodeIds }: { edgeIds: string[]; nodeIds: string[] }) => {
      const requestedNodeIds = new Set(nodeIds);
      const existingNodeIds = projectGraph.nodes.filter((node) => requestedNodeIds.has(node.id)).map((node) => node.id);

      if (existingNodeIds.length > 0) {
        editGraph({ nodeIds: existingNodeIds, type: 'removeNodes' });
      }

      const removedNodeIds = new Set(existingNodeIds);
      const requestedEdgeIds = new Set(edgeIds);
      const existingEdgeIds = projectGraph.edges
        .filter(
          (edge) =>
            requestedEdgeIds.has(edge.id) && !removedNodeIds.has(edge.source) && !removedNodeIds.has(edge.target)
        )
        .map((edge) => edge.id);

      if (existingEdgeIds.length > 0) {
        editGraph({ edgeIds: existingEdgeIds, type: 'removeEdges' });
      }
    },
    [editGraph, projectGraph.edges, projectGraph.nodes]
  );
  const { eraserHandlers, eraserOverlay } = useEraser({
    enabled: tool === 'eraser',
    flowInstance,
    onErase: eraseElements,
  });
  const pointerToolHandlers = tool === 'lasso' ? lassoHandlers : tool === 'eraser' ? eraserHandlers : {};
  const renderedFlowModel = useMemo(
    () =>
      flowModel && isLargeGraph && !isFullGraphMounted
        ? getInitialRenderFlowModel(flowModel, defaultViewport)
        : flowModel,
    [defaultViewport, flowModel, isFullGraphMounted, isLargeGraph]
  );
  const renderedFlowNodes = renderedFlowModel?.nodes ?? EMPTY_FLOW_NODES;
  const renderedFlowEdges = renderedFlowModel?.edges ?? EMPTY_FLOW_EDGES;

  useEffect(() => {
    markWorkbenchPerf(perfMountMarkRef.current, perfSource);
  }, [markWorkbenchPerf, perfSource]);

  useEffect(() => {
    if (isPreparing) {
      return;
    }

    markWorkbenchPerf(perfReadyMarkRef.current, perfSource);
    measureWorkbenchPerf(
      'workflow:editor-mounted-to-ready',
      perfMountMarkRef.current,
      perfSource,
      perfReadyMarkRef.current
    );
  }, [isPreparing, markWorkbenchPerf, measureWorkbenchPerf, perfSource]);

  useEffect(() => {
    if (!isFullGraphMounted || !isLargeGraph) {
      return;
    }

    markWorkbenchPerf('workflow:full-graph-mounted', perfSource);
    measureWorkbenchPerf(
      'workflow:full-graph-expand',
      'workflow:full-graph-expand-start',
      perfSource,
      'workflow:full-graph-mounted'
    );
  }, [isFullGraphMounted, isLargeGraph, markWorkbenchPerf, measureWorkbenchPerf, perfSource]);

  useEffect(() => {
    if (!workflowShowMinimap || !isLargeGraph || isPreparing) {
      return undefined;
    }

    const timeoutId = window.setTimeout(() => {
      markWorkbenchPerf('workflow:minimap-ready', perfSource);
      setIsMinimapReady(true);
    }, WORKFLOW_MINIMAP_DELAY_MS);

    return () => window.clearTimeout(timeoutId);
  }, [isLargeGraph, isPreparing, markWorkbenchPerf, perfSource, workflowShowMinimap]);

  useEffect(() => {
    const pendingSelection = pendingSelectionRef.current;

    pendingSelectionRef.current = null;
    const nextSelectedNodeIds = pendingSelection ?? selectedNodeIds;
    const nextSelectedNodeIdSet = new Set(nextSelectedNodeIds);
    let frameId: number | null = null;

    const lastBuiltModel = lastBuiltModelRef.current;

    if (
      pendingSelection === null &&
      !shouldDeferInitialBuild &&
      lastBuiltModel?.projectGraph === projectGraph &&
      lastBuiltModel.edgeType === edgeType &&
      lastBuiltModel.invocationTemplates === invocationTemplates &&
      lastBuiltModel.reduceMotion === reduceMotion &&
      lastBuiltModel.selectedNodeIds === nextSelectedNodeIds
    ) {
      return undefined;
    }

    const updateModel = () => {
      lastBuiltModelRef.current = {
        edgeType,
        invocationTemplates,
        projectGraph,
        reduceMotion,
        selectedNodeIds: nextSelectedNodeIds,
      };

      startTransition(() => {
        setFlowModel((current) => {
          const next = buildWorkflowFlowModel({
            document: projectGraph,
            edgeType,
            canUseCache,
            isCompact: isLargeGraph,
            invocationTemplates,
            perfSource,
            previousEdges: current?.edges,
            previousNodes: current?.nodes,
            reduceMotion,
            selectedNodeIds: nextSelectedNodeIdSet,
            time: timeWorkbenchPerf,
          });

          return { ...next, nodes: withNodeSelection(next.nodes, nextSelectedNodeIdSet) };
        });
      });
    };

    if (shouldDeferInitialBuild) {
      frameId = window.requestAnimationFrame(updateModel);
    } else {
      updateModel();
    }

    if (pendingSelection) {
      reportNodeSelection(pendingSelection);
    }

    return () => {
      if (frameId !== null) {
        window.cancelAnimationFrame(frameId);
      }
    };
  }, [
    canUseCache,
    edgeType,
    invocationTemplates,
    isLargeGraph,
    perfSource,
    projectGraph,
    reduceMotion,
    selectedNodeIds,
    shouldDeferInitialBuild,
    timeWorkbenchPerf,
  ]);

  // Expose the instance to the widget header's actions (outside this provider).
  useEffect(() => {
    if (!flowInstance) {
      return undefined;
    }

    registerWorkflowFlowInstance(flowInstance);

    return () => {
      releaseWorkflowFlowInstance(flowInstance);
    };
  }, [flowInstance]);

  const applyRequestedSelection = useEffectEvent(() => {
    const selectionRequest = workflowSelectionStore.getSnapshot().selectionRequest;

    if (!selectionRequest || !flowInstance) {
      return;
    }

    selectNodes(selectionRequest.nodeIds);
    void flowInstance.fitView({
      duration: reduceMotion ? 0 : 300,
      maxZoom: 1.25,
      nodes: selectionRequest.nodeIds.map((id) => ({ id })),
    });
    clearNodeSelectionRequest();
  });

  // Outside surfaces publish selection requests through an external store.
  // Subscribe at that seam so the request does not need an intermediate
  // render-and-effect cycle before it reaches the flow instance.
  useEffect(() => {
    const pendingRequestTimer = window.setTimeout(applyRequestedSelection, 0);
    const unsubscribe = workflowSelectionStore.subscribe(applyRequestedSelection);

    return () => {
      window.clearTimeout(pendingRequestTimer);
      unsubscribe();
    };
  }, [flowInstance]);

  /** The context-menu node acts alone unless it is part of the current selection. */
  const getActionNodeIds = useCallback(
    (nodeId?: string): string[] => {
      const selectedIds = flowNodes.filter((node) => node.selected).map((node) => node.id);

      if (nodeId && !selectedIds.includes(nodeId)) {
        return [nodeId];
      }

      return selectedIds;
    },
    [flowNodes]
  );

  const copyNodes = useCallback(
    (nodeId?: string) => {
      const copiedCount = copyNodesToClipboard(projectGraph, getActionNodeIds(nodeId));

      if (copiedCount > 0) {
        notify.success(`Copied ${copiedCount} node${copiedCount === 1 ? '' : 's'}`);
      }
    },
    [getActionNodeIds, notify, projectGraph]
  );

  const pasteNodes = useCallback(() => {
    const { edges, nodes } = buildPasteElements();

    if (nodes.length === 0) {
      return;
    }

    pendingSelectionRef.current = nodes.map((node) => node.id);
    editGraph({ edges, nodes, type: 'addGraphElements' });
  }, [editGraph]);

  const duplicateNodes = useCallback(
    (nodeId?: string) => {
      const { edges, nodes } = buildDuplicateElements(projectGraph, getActionNodeIds(nodeId));

      if (nodes.length === 0) {
        return;
      }

      pendingSelectionRef.current = nodes.map((node) => node.id);
      editGraph({ edges, nodes, type: 'addGraphElements' });
    },
    [editGraph, getActionNodeIds, projectGraph]
  );

  const deleteNodes = useCallback(
    (nodeId?: string) => {
      const nodeIds = getActionNodeIds(nodeId);

      if (nodeIds.length > 0) {
        editGraph({ nodeIds, type: 'removeNodes' });
      }
    },
    [editGraph, getActionNodeIds]
  );

  const addConnector = useCallback(
    (position: XYPosition) => {
      editGraph({ node: buildConnectorNode(position), type: 'addNode' });
    },
    [editGraph]
  );

  const selectAll = () => {
    selectNodes(projectGraph.nodes.map((node) => node.id));
  };

  const deleteSelection = () => {
    const nodeIds = flowNodes.filter((node) => node.selected).map((node) => node.id);
    const edgeIds = flowEdges.filter((edge) => edge.selected).map((edge) => edge.id);

    if (nodeIds.length > 0) {
      editGraph({ nodeIds, type: 'removeNodes' });
    }

    if (edgeIds.length > 0) {
      editGraph({ edgeIds, type: 'removeEdges' });
    }
  };

  const executeWorkflowHotkey = useEffectEvent((commandId: string) => {
    switch (commandId) {
      case 'workflows.addNode': {
        setAddNodeOpen(true);
        return;
      }
      case 'workflows.copySelection': {
        copyNodes();
        return;
      }
      case 'workflows.pasteSelection':
      case 'workflows.pasteSelectionWithEdges': {
        pasteNodes();
        return;
      }
      case 'workflows.duplicateSelection': {
        duplicateNodes();
        return;
      }
      case 'workflows.selectAll': {
        selectAll();
        return;
      }
      case 'workflows.deleteSelection': {
        deleteSelection();
        return;
      }
      case 'workflows.undo': {
        undo();
        return;
      }
      case 'workflows.redo': {
        redo();
        return;
      }
    }
  });

  useEffect(() => {
    const hotkeys = [
      ['workflows.addNode', 'Add workflow node', ['shift+a', 'space']],
      ['workflows.copySelection', 'Copy workflow selection', ['mod+c']],
      ['workflows.pasteSelection', 'Paste workflow selection', ['mod+v']],
      ['workflows.pasteSelectionWithEdges', 'Paste workflow selection with edges', ['mod+shift+v']],
      ['workflows.duplicateSelection', 'Duplicate workflow selection', ['mod+d']],
      ['workflows.selectAll', 'Select all workflow nodes', ['mod+a']],
      ['workflows.deleteSelection', 'Delete workflow selection', ['delete', 'backspace']],
      ['workflows.undo', 'Undo workflow edit', ['mod+z']],
      ['workflows.redo', 'Redo workflow edit', ['mod+shift+z', 'mod+y']],
    ] as const;
    const disposers = hotkeys.flatMap(([id, title, defaultKeys]) => [
      runtime.commands.register({ handler: () => executeWorkflowHotkey(id), id, title }),
      runtime.hotkeys.register({ commandId: id, defaultKeys: [...defaultKeys], id, title }),
    ]);

    return () => {
      disposers.forEach((dispose) => dispose());
    };
  }, [runtime.commands, runtime.hotkeys]);

  const onNodesChange = useCallback(
    (changes: NodeChange<WorkflowFlowNode>[]) => {
      const removedNodeIds = changes.flatMap((change) => (change.type === 'remove' ? [change.id] : []));

      if (removedNodeIds.length > 0) {
        editGraph({ nodeIds: removedNodeIds, type: 'removeNodes' });
      }

      for (const change of changes) {
        if (change.type === 'position' && change.dragging === false && change.position) {
          editGraph({ nodeId: change.id, position: change.position, type: 'setNodePosition' });
        }
      }

      setFlowModel((current) => (current ? { ...current, nodes: applyNodeChanges(changes, current.nodes) } : current));
    },
    [editGraph]
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange<WorkflowFlowEdge>[]) => {
      const removedEdgeIds = changes.flatMap((change) => (change.type === 'remove' ? [change.id] : []));

      if (removedEdgeIds.length > 0) {
        editGraph({ edgeIds: removedEdgeIds, type: 'removeEdges' });
      }

      setFlowModel((current) => (current ? { ...current, edges: applyEdgeChanges(changes, current.edges) } : current));
    },
    [editGraph]
  );

  const isValidConnection: IsValidConnection<WorkflowFlowEdge> = useCallback(
    (connection) => {
      if (!connection.sourceHandle || !connection.targetHandle) {
        return false;
      }

      // Lenient wiring: the user opted out of type validation in settings.
      if (!workflowValidateConnections) {
        return true;
      }

      return (
        validateConnection(
          {
            sourceHandle: connection.sourceHandle,
            sourceNodeId: connection.source,
            targetHandle: connection.targetHandle,
            targetNodeId: connection.target,
          },
          projectGraph,
          templates
        ) === null
      );
    },
    [projectGraph, templates, workflowValidateConnections]
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      if (!connection.sourceHandle || !connection.targetHandle) {
        return;
      }

      editGraph({
        edge: {
          id: createWorkflowId('edge'),
          source: connection.source,
          sourceHandle: connection.sourceHandle,
          target: connection.target,
          targetHandle: connection.targetHandle,
          type: 'default',
        },
        type: 'addEdge',
      });
    },
    [editGraph]
  );

  const onConnectEnd = useCallback<OnConnectEnd>(
    (event, connectionState) => {
      if (!flowInstance || !(event.target instanceof Element) || !event.target.closest('.react-flow__pane')) {
        return;
      }

      if (connectionState.isValid || connectionState.toHandle || connectionState.toNode) {
        return;
      }

      const handle = connectionState.fromHandle;
      const handleId = handle?.id;
      const nodeId = handle?.nodeId ?? connectionState.fromNode?.id;

      if (!handleId || !nodeId) {
        return;
      }

      const position = getEventClientPosition(event);

      if (!position) {
        return;
      }

      if (handle.type === 'source') {
        const sourceType = getWorkflowSourceFieldType(projectGraph, templates, nodeId, handleId);

        if (sourceType === undefined) {
          return;
        }

        setAddNodeOpen(true, flowInstance.screenToFlowPosition(position), {
          kind: 'source',
          sourceHandle: handleId,
          sourceNodeId: nodeId,
          sourceType,
        });
        return;
      }

      if (handle.type === 'target') {
        const targetType = getWorkflowTargetFieldType(projectGraph, templates, nodeId, handleId);

        if (targetType === undefined) {
          return;
        }

        setAddNodeOpen(true, flowInstance.screenToFlowPosition(position), {
          kind: 'target',
          targetHandle: handleId,
          targetNodeId: nodeId,
          targetType,
        });
      }
    },
    [flowInstance, projectGraph, templates]
  );

  const onNodeContextMenu = useCallback((event: ReactMouseEvent, node: WorkflowFlowNode) => {
    event.preventDefault();
    setContextMenu({
      kind: 'node',
      isNodeOpen: node.type === 'invocation' ? node.data.documentNode.data.isOpen : null,
      nodeId: node.id,
      x: event.clientX,
      y: event.clientY,
    });
  }, []);

  const onPaneContextMenu = useCallback(
    (event: ReactMouseEvent | globalThis.MouseEvent) => {
      if (!flowInstance) {
        return;
      }

      event.preventDefault();
      if (
        event.target instanceof Element &&
        event.target.closest('.react-flow__nodesselection') &&
        getActionNodeIds().length > 0
      ) {
        setContextMenu({
          kind: 'node',
          isNodeOpen: null,
          x: event.clientX,
          y: event.clientY,
        });
        return;
      }

      setContextMenu({
        kind: 'pane',
        position: flowInstance.screenToFlowPosition({ x: event.clientX, y: event.clientY }),
        x: event.clientX,
        y: event.clientY,
      });
    },
    [flowInstance, getActionNodeIds]
  );

  const onEditorContextMenuCapture = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      if (
        event.target instanceof Element &&
        event.target.closest('.react-flow__nodesselection') &&
        getActionNodeIds().length > 0
      ) {
        event.preventDefault();
        event.stopPropagation();
        setContextMenu({
          kind: 'node',
          isNodeOpen: null,
          x: event.clientX,
          y: event.clientY,
        });
      }
    },
    [getActionNodeIds]
  );

  const onNodeClick = useCallback(
    (_: ReactMouseEvent, node: WorkflowFlowNode) => {
      if (tool === 'eraser') {
        eraseElements({ edgeIds: [], nodeIds: [node.id] });
      }
    },
    [eraseElements, tool]
  );

  const onEdgeClick = useCallback(
    (_: ReactMouseEvent, edge: WorkflowFlowEdge) => {
      if (tool === 'eraser') {
        eraseElements({ edgeIds: [edge.id], nodeIds: [] });
      }
    },
    [eraseElements, tool]
  );

  const onSelectionChange = useCallback(
    ({ nodes }: { nodes: WorkflowFlowNode[] }) => {
      const nodeIds = nodes.map((node) => node.id);

      setFlowModel((current) => {
        if (!current) {
          return current;
        }

        const index = createWorkflowGraphIndex(projectGraph.nodes, projectGraph.edges);

        return {
          ...current,
          edges: toFlowEdges(
            projectGraph,
            current.edges,
            edgeType,
            getSelectedNodeIdSet(nodes),
            invocationTemplates,
            reduceMotion,
            index
          ),
        };
      });
      reportNodeSelection(nodeIds);
    },
    [edgeType, invocationTemplates, projectGraph, reduceMotion]
  );

  const onNodeMouseEnter = useCallback((_: ReactMouseEvent, node: WorkflowFlowNode) => {
    reportNodeHover(node.id);
  }, []);

  const onNodeMouseLeave = useCallback((_: ReactMouseEvent, node: WorkflowFlowNode) => {
    if (workflowSelectionStore.getSnapshot().hoveredNodeId === node.id) {
      reportNodeHover(null);
    }
  }, []);
  const onMoveEnd = useCallback(
    (_: MouseEvent | TouchEvent | null, viewport: Viewport) => {
      setWorkflowViewport(viewportKey, viewport);
    },
    [viewportKey]
  );
  const onFlowInit = useCallback(
    (instance: WorkflowFlowInstance) => {
      const flowInitMark = 'workflow:react-flow-init';

      markWorkbenchPerf(flowInitMark, perfSource);
      measureWorkbenchPerf(
        'workflow:editor-ready-to-react-flow-init',
        perfReadyMarkRef.current,
        perfSource,
        flowInitMark
      );
      measureWorkbenchPerf(
        'workflow:editor-mounted-to-react-flow-init',
        perfMountMarkRef.current,
        perfSource,
        flowInitMark
      );
      setFlowInstance(instance);

      if (isLargeGraph && !isFullGraphMounted && !hasScheduledFullGraphMountRef.current) {
        hasScheduledFullGraphMountRef.current = true;

        window.setTimeout(() => {
          markWorkbenchPerf('workflow:full-graph-expand-start', perfSource);
          setIsFullGraphMounted(true);
        }, 0);
      }
    },
    [isFullGraphMounted, isLargeGraph, markWorkbenchPerf, measureWorkbenchPerf, perfSource]
  );

  const editorCss = useMemo(
    () => ({
      ...flowThemeCss,
      '& .react-flow__node': { opacity: nodeOpacity },
      '& .react-flow__edge .react-flow__edge-path': {
        opacity: 0.38,
        transition: 'opacity var(--wb-motion-duration-fast) ease',
      },
      '& .react-flow__edge.selected .react-flow__edge-path, & .react-flow__edge.workflow-selected-node-edge .react-flow__edge-path, & .react-flow__edge:hover .react-flow__edge-path':
        {
          opacity: 1,
        },
      '& .react-flow__edge.workflow-selected-node-edge .react-flow__edge-path': {
        animation: 'dashdraw var(--wb-motion-duration-slow) linear var(--wb-motion-animation-iteration-count)',
        strokeDasharray: '5',
      },
      ...(tool === 'eraser'
        ? { '& .react-flow__edge, & .react-flow__node, & .react-flow__pane': { cursor: 'crosshair' } }
        : {}),
      ...(tool === 'lasso' ? { '& .react-flow__pane': { cursor: 'crosshair' } } : {}),
    }),
    [nodeOpacity, tool]
  );
  const panOnDrag = useMemo(() => (tool === 'pan' ? true : [1, 2]), [tool]);
  const proOptions = useMemo(() => ({ hideAttribution: true }), []);
  const flowStyle = useMemo(() => ({ background: 'transparent' }), []);
  const onContextMenuAddConnector = useCallback(
    (position: XYPosition) => {
      addConnector(position);
      setContextMenu(null);
    },
    [addConnector]
  );
  const onContextMenuClose = useCallback(() => setContextMenu(null), []);
  const onContextMenuCopy = useCallback(() => {
    copyNodes(contextMenu?.kind === 'node' ? contextMenu.nodeId : undefined);
    setContextMenu(null);
  }, [contextMenu, copyNodes]);
  const onContextMenuDelete = useCallback(() => {
    deleteNodes(contextMenu?.kind === 'node' ? contextMenu.nodeId : undefined);
    setContextMenu(null);
  }, [contextMenu, deleteNodes]);
  const onContextMenuDuplicate = useCallback(() => {
    duplicateNodes(contextMenu?.kind === 'node' ? contextMenu.nodeId : undefined);
    setContextMenu(null);
  }, [contextMenu, duplicateNodes]);
  const onContextMenuPaste = useCallback(() => {
    pasteNodes();
    setContextMenu(null);
  }, [pasteNodes]);
  const onContextMenuToggleOpen = useCallback(() => {
    if (contextMenu?.kind === 'node' && contextMenu.nodeId && contextMenu.isNodeOpen !== null) {
      editGraph({ isOpen: !contextMenu.isNodeOpen, nodeId: contextMenu.nodeId, type: 'setNodeIsOpen' });
    }

    setContextMenu(null);
  }, [contextMenu, editGraph]);

  if (isPreparing) {
    return <WorkflowEditorPreparingState edgeCount={projectGraph.edges.length} nodeCount={projectGraph.nodes.length} />;
  }

  return (
    <Box
      bg="bg.inset"
      css={editorCss}
      h="full"
      position="relative"
      w="full"
      onContextMenuCapture={onEditorContextMenuCapture}
      {...pointerToolHandlers}
    >
      <ReactFlow<WorkflowFlowNode, WorkflowFlowEdge>
        key={viewportKey}
        colorMode={getFlowColorMode(themeId)}
        connectionLineType={edgeType === 'step' ? ConnectionLineType.Step : ConnectionLineType.Bezier}
        defaultEdgeOptions={DEFAULT_EDGE_OPTIONS}
        defaultViewport={defaultViewport}
        deleteKeyCode={DELETE_KEY_CODES}
        elevateEdgesOnSelect
        edges={renderedFlowEdges}
        edgeTypes={edgeTypes}
        isValidConnection={isValidConnection}
        maxZoom={2}
        minZoom={0.1}
        nodes={renderedFlowNodes}
        nodeTypes={nodeTypes}
        onlyRenderVisibleElements={isLargeGraph}
        panOnDrag={panOnDrag}
        proOptions={proOptions}
        selectionMode={SelectionMode.Partial}
        selectionOnDrag={tool === 'box-select'}
        snapGrid={SNAP_GRID}
        snapToGrid={workflowSnapToGrid || isSnapHeld}
        style={flowStyle}
        onConnect={onConnect}
        onConnectEnd={onConnectEnd}
        onEdgeClick={onEdgeClick}
        onEdgesChange={onEdgesChange}
        onInit={onFlowInit}
        onNodeClick={onNodeClick}
        onNodeContextMenu={onNodeContextMenu}
        onNodeMouseEnter={onNodeMouseEnter}
        onNodeMouseLeave={onNodeMouseLeave}
        onNodesChange={onNodesChange}
        onMoveEnd={onMoveEnd}
        onPaneContextMenu={onPaneContextMenu}
        onSelectionChange={onSelectionChange}
      >
        <Background
          bgColor="var(--xy-background-color)"
          color="var(--wb-flow-grid)"
          gap={24}
          id={`workflow-grid-${backgroundId}`}
          size={1.5}
          variant={BackgroundVariant.Dots}
        />
        <EditorToolbar
          nodeOpacity={nodeOpacity}
          tool={tool}
          onNodeOpacityChange={setNodeOpacity}
          onToolChange={setTool}
        />
        {workflowShowMinimap && (!isLargeGraph || isMinimapReady) ? <FlowMiniMap /> : null}
      </ReactFlow>
      {lassoOverlay}
      {eraserOverlay}
      <NodeContextMenu
        canPaste={hasClipboardNodes}
        menuState={contextMenu}
        onAddConnector={onContextMenuAddConnector}
        onClose={onContextMenuClose}
        onCopy={onContextMenuCopy}
        onDelete={onContextMenuDelete}
        onDuplicate={onContextMenuDuplicate}
        onPaste={onContextMenuPaste}
        onToggleOpen={onContextMenuToggleOpen}
      />
    </Box>
  );
};

const ReadinessBanner = () => {
  const projectGraph = useWorkflowProjectSelector((project) => project.projectGraph);
  const templatesStatus = useInvocationTemplatesSelector((snapshot) => snapshot.status);
  const templates = useInvocationTemplatesSelector((snapshot) => snapshot.templates);
  const templatesSnapshot = useMemo(
    () => ({ error: null, status: templatesStatus, templates }),
    [templatesStatus, templates]
  );
  const readiness = useMemo(
    () => getProjectGraphReadiness(projectGraph, templatesSnapshot),
    [projectGraph, templatesSnapshot]
  );

  if (readiness.canInvoke || projectGraph.nodes.length === 0) {
    return null;
  }

  return (
    <Box
      bg="bg.muted"
      borderColor="border.subtle"
      borderRadius="md"
      borderWidth="1px"
      left="3"
      maxW="24rem"
      p="2"
      position="absolute"
      top="3"
      zIndex="1"
    >
      <Stack gap="0.5">
        {readiness.reasons.slice(0, 4).map((reason, index) => (
          <Text key={`${index}:${reason}`} color="fg.muted" fontSize="2xs">
            {reason}
          </Text>
        ))}
        {readiness.reasons.length > 4 ? (
          <Text color="fg.subtle" fontSize="2xs">
            +{readiness.reasons.length - 4} more
          </Text>
        ) : null}
      </Stack>
    </Box>
  );
};

export const WorkflowEditorView = ({ runtime }: { runtime: WorkflowRuntimeApi }) => {
  const flowIdentity = useWorkflowProjectSelector(
    (project) =>
      `${project.id}:${isLargeWorkflowGraph({ edgeCount: project.projectGraph.edges.length, nodeCount: project.projectGraph.nodes.length }) ? 'large' : 'standard'}`
  );

  useEffect(() => {
    ensureInvocationTemplatesLoaded();
  }, []);

  return (
    <ReactFlowProvider>
      <Flex direction="column" h="full" minH="0" w="full">
        <Box flex="1" minH="0" position="relative">
          <WorkflowFlow key={flowIdentity} runtime={runtime} />
          <ReadinessBanner />
        </Box>
      </Flex>
    </ReactFlowProvider>
  );
};
