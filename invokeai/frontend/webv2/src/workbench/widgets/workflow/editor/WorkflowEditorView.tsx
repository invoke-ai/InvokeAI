import type { XYPosition } from '@workbench/workflows/types';

import { Box, Flex, Stack, Text } from '@chakra-ui/react';
import { FlowMiniMap, flowThemeCss, getFlowColorMode } from '@workbench/graph-preview';
import '@xyflow/react/dist/style.css';
import { useWorkbenchPreferences } from '@workbench/settings/store';
import { useNotify } from '@workbench/useNotify';
import { setAddNodeOpen } from '@workbench/widgets/workflow/workflowUiStore';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { getProjectGraphReadiness } from '@workbench/workflows/buildGraph';
import { buildConnectorNode, createWorkflowId } from '@workbench/workflows/document';
import { ensureInvocationTemplatesLoaded, useInvocationTemplatesSnapshot } from '@workbench/workflows/templates';
import {
  getWorkflowSourceFieldType,
  getWorkflowTargetFieldType,
  validateConnection,
} from '@workbench/workflows/validation';
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
} from '@xyflow/react';
import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
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
  clearNodeSelectionRequest,
  reportNodeHover,
  reportNodeSelection,
  workflowSelectionStore,
} from './selectionStore';
import { useEraser } from './useEraser';
import { useLasso } from './useLasso';
import { useModifierHeld } from './useModifierHeld';
import { WorkflowEdge } from './WorkflowEdge';

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

const isEditableTarget = (target: EventTarget | null): boolean =>
  target instanceof HTMLElement && target.closest('input, textarea, select, [contenteditable="true"]') !== null;

const getEventClientPosition = (event: MouseEvent | TouchEvent): { x: number; y: number } | null => {
  if (event instanceof MouseEvent) {
    return { x: event.clientX, y: event.clientY };
  }

  const touch = event.changedTouches[0];

  return touch ? { x: touch.clientX, y: touch.clientY } : null;
};

const getSelectedNodeIdSet = (nodes: WorkflowFlowNode[]): Set<string> =>
  new Set(nodes.filter((node) => node.selected).map((node) => node.id));

const WorkflowFlow = () => {
  const projectGraph = useActiveProjectSelector((project) => project.projectGraph);
  const dispatch = useWorkbenchDispatch();
  const notify = useNotify();
  const { themeId, workflowEdgeStyle, workflowShowMinimap, workflowSnapToGrid, workflowValidateConnections } =
    useWorkbenchPreferences();
  const templatesSnapshot = useInvocationTemplatesSnapshot();
  const invocationTemplates = templatesSnapshot.status === 'loaded' ? templatesSnapshot.templates : undefined;
  const edgeType: FlowEdgeType = workflowEdgeStyle === 'square' ? 'step' : 'default';
  const [flowNodes, setFlowNodes] = useState<WorkflowFlowNode[]>(() =>
    toFlowNodes(projectGraph, [], invocationTemplates)
  );
  const [flowEdges, setFlowEdges] = useState<WorkflowFlowEdge[]>(() =>
    toFlowEdges(projectGraph, [], edgeType, new Set<string>(), invocationTemplates)
  );
  const [flowInstance, setFlowInstance] = useState<WorkflowFlowInstance | null>(null);
  const [tool, setTool] = useState<EditorTool>('pan');
  const [nodeOpacity, setNodeOpacity] = useState(1);
  const [contextMenu, setContextMenu] = useState<WorkflowContextMenuState | null>(null);
  const { selectedNodeIds, selectionRequest } = workflowSelectionStore.useSnapshot();
  const isSnapHeld = useModifierHeld('Control');
  const hasClipboardNodes = useHasClipboardNodes();
  const backgroundId = useId().replace(/:/g, '');
  /** Node ids to select once the next document-driven rebuild lands (fresh paste/duplicate results). */
  const pendingSelectionRef = useRef<string[] | null>(null);

  const selectNodes = useCallback(
    (nodeIds: string[]) => {
      const selectedNodeIdSet = new Set(nodeIds);

      setFlowNodes((current) => withNodeSelection(current, selectedNodeIdSet));
      setFlowEdges((current) => toFlowEdges(projectGraph, current, edgeType, selectedNodeIdSet, invocationTemplates));
      reportNodeSelection(nodeIds);
    },
    [edgeType, invocationTemplates, projectGraph]
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
        dispatch({ action: { nodeIds: existingNodeIds, type: 'removeNodes' }, type: 'applyProjectGraphAction' });
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
        dispatch({ action: { edgeIds: existingEdgeIds, type: 'removeEdges' }, type: 'applyProjectGraphAction' });
      }
    },
    [dispatch, projectGraph.edges, projectGraph.nodes]
  );
  const { eraserHandlers, eraserOverlay } = useEraser({
    enabled: tool === 'eraser',
    flowInstance,
    onErase: eraseElements,
  });
  const pointerToolHandlers = tool === 'lasso' ? lassoHandlers : tool === 'eraser' ? eraserHandlers : {};

  useEffect(() => {
    const pendingSelection = pendingSelectionRef.current;

    pendingSelectionRef.current = null;
    const nextSelectedNodeIds = pendingSelection ?? selectedNodeIds;
    const nextSelectedNodeIdSet = new Set(nextSelectedNodeIds);

    setFlowNodes((current) => {
      const next = toFlowNodes(projectGraph, current, invocationTemplates);

      return pendingSelection ? withNodeSelection(next, nextSelectedNodeIdSet) : next;
    });
    setFlowEdges((current) => toFlowEdges(projectGraph, current, edgeType, nextSelectedNodeIdSet, invocationTemplates));

    if (pendingSelection) {
      reportNodeSelection(pendingSelection);
    }
  }, [edgeType, invocationTemplates, projectGraph, selectedNodeIds]);

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

  // Outside surfaces (the form builder's zoom-to-node) request a selection;
  // the editor applies it and brings the nodes into view.
  useEffect(() => {
    if (!selectionRequest || !flowInstance) {
      return;
    }

    selectNodes(selectionRequest.nodeIds);
    void flowInstance.fitView({
      duration: 300,
      maxZoom: 1.25,
      nodes: selectionRequest.nodeIds.map((id) => ({ id })),
    });
    clearNodeSelectionRequest();
  }, [flowInstance, selectNodes, selectionRequest]);

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

  const copyNodes = (nodeId?: string) => {
    const copiedCount = copyNodesToClipboard(projectGraph, getActionNodeIds(nodeId));

    if (copiedCount > 0) {
      notify.success(`Copied ${copiedCount} node${copiedCount === 1 ? '' : 's'}`);
    }
  };

  const pasteNodes = () => {
    const { edges, nodes } = buildPasteElements();

    if (nodes.length === 0) {
      return;
    }

    pendingSelectionRef.current = nodes.map((node) => node.id);
    dispatch({ action: { edges, nodes, type: 'addGraphElements' }, type: 'applyProjectGraphAction' });
  };

  const duplicateNodes = (nodeId?: string) => {
    const { edges, nodes } = buildDuplicateElements(projectGraph, getActionNodeIds(nodeId));

    if (nodes.length === 0) {
      return;
    }

    pendingSelectionRef.current = nodes.map((node) => node.id);
    dispatch({ action: { edges, nodes, type: 'addGraphElements' }, type: 'applyProjectGraphAction' });
  };

  const deleteNodes = (nodeId?: string) => {
    const nodeIds = getActionNodeIds(nodeId);

    if (nodeIds.length > 0) {
      dispatch({ action: { nodeIds, type: 'removeNodes' }, type: 'applyProjectGraphAction' });
    }
  };

  const addConnector = (position: XYPosition) => {
    dispatch({ action: { node: buildConnectorNode(position), type: 'addNode' }, type: 'applyProjectGraphAction' });
  };

  const onEditorKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (isEditableTarget(event.target) || !(event.ctrlKey || event.metaKey)) {
      return;
    }

    const key = event.key.toLowerCase();

    if (key === 'c') {
      copyNodes();
    } else if (key === 'v') {
      pasteNodes();
    } else if (key === 'd') {
      event.preventDefault();
      duplicateNodes();
    }
  };

  const onNodesChange = useCallback(
    (changes: NodeChange<WorkflowFlowNode>[]) => {
      const removedNodeIds = changes.flatMap((change) => (change.type === 'remove' ? [change.id] : []));

      if (removedNodeIds.length > 0) {
        dispatch({ action: { nodeIds: removedNodeIds, type: 'removeNodes' }, type: 'applyProjectGraphAction' });
      }

      for (const change of changes) {
        if (change.type === 'position' && change.dragging === false && change.position) {
          dispatch({
            action: { nodeId: change.id, position: change.position, type: 'setNodePosition' },
            type: 'applyProjectGraphAction',
          });
        }
      }

      setFlowNodes((current) => applyNodeChanges(changes, current));
    },
    [dispatch]
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange<WorkflowFlowEdge>[]) => {
      const removedEdgeIds = changes.flatMap((change) => (change.type === 'remove' ? [change.id] : []));

      if (removedEdgeIds.length > 0) {
        dispatch({ action: { edgeIds: removedEdgeIds, type: 'removeEdges' }, type: 'applyProjectGraphAction' });
      }

      setFlowEdges((current) => applyEdgeChanges(changes, current));
    },
    [dispatch]
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
          templatesSnapshot.templates
        ) === null
      );
    },
    [projectGraph, templatesSnapshot.templates, workflowValidateConnections]
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      if (!connection.sourceHandle || !connection.targetHandle) {
        return;
      }

      dispatch({
        action: {
          edge: {
            id: createWorkflowId('edge'),
            source: connection.source,
            sourceHandle: connection.sourceHandle,
            target: connection.target,
            targetHandle: connection.targetHandle,
            type: 'default',
          },
          type: 'addEdge',
        },
        type: 'applyProjectGraphAction',
      });
    },
    [dispatch]
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
        const sourceType = getWorkflowSourceFieldType(projectGraph, templatesSnapshot.templates, nodeId, handleId);

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
        const targetType = getWorkflowTargetFieldType(projectGraph, templatesSnapshot.templates, nodeId, handleId);

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
    [flowInstance, projectGraph, templatesSnapshot.templates]
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

      setFlowEdges((current) =>
        toFlowEdges(projectGraph, current, edgeType, getSelectedNodeIdSet(nodes), invocationTemplates)
      );
      reportNodeSelection(nodeIds);
    },
    [edgeType, invocationTemplates, projectGraph]
  );

  const onNodeMouseEnter = useCallback((_: ReactMouseEvent, node: WorkflowFlowNode) => {
    reportNodeHover(node.id);
  }, []);

  const onNodeMouseLeave = useCallback((_: ReactMouseEvent, node: WorkflowFlowNode) => {
    if (workflowSelectionStore.getSnapshot().hoveredNodeId === node.id) {
      reportNodeHover(null);
    }
  }, []);

  const editorCss = useMemo(
    () => ({
      ...flowThemeCss,
      '& .react-flow__node': { opacity: nodeOpacity },
      '& .react-flow__edge .react-flow__edge-path': {
        opacity: 0.38,
        transition: 'opacity 0.12s ease',
      },
      '& .react-flow__edge.selected .react-flow__edge-path, & .react-flow__edge.workflow-selected-node-edge .react-flow__edge-path, & .react-flow__edge:hover .react-flow__edge-path':
        {
          opacity: 1,
        },
      '& .react-flow__edge.workflow-selected-node-edge .react-flow__edge-path': {
        animation: 'dashdraw 0.5s linear infinite',
        strokeDasharray: '5',
      },
      ...(tool === 'eraser'
        ? { '& .react-flow__edge, & .react-flow__node, & .react-flow__pane': { cursor: 'crosshair' } }
        : {}),
      ...(tool === 'lasso' ? { '& .react-flow__pane': { cursor: 'crosshair' } } : {}),
    }),
    [nodeOpacity, tool]
  );

  return (
    <Box
      bg="bg.inset"
      css={editorCss}
      h="full"
      position="relative"
      w="full"
      onContextMenuCapture={onEditorContextMenuCapture}
      onKeyDown={onEditorKeyDown}
      {...pointerToolHandlers}
    >
      <ReactFlow<WorkflowFlowNode, WorkflowFlowEdge>
        colorMode={getFlowColorMode(themeId)}
        connectionLineType={edgeType === 'step' ? ConnectionLineType.Step : ConnectionLineType.Bezier}
        defaultEdgeOptions={DEFAULT_EDGE_OPTIONS}
        deleteKeyCode={DELETE_KEY_CODES}
        elevateEdgesOnSelect
        edges={flowEdges}
        edgeTypes={edgeTypes}
        fitView
        isValidConnection={isValidConnection}
        maxZoom={2}
        minZoom={0.1}
        nodes={flowNodes}
        nodeTypes={nodeTypes}
        panOnDrag={tool === 'pan' ? true : [1, 2]}
        proOptions={{ hideAttribution: true }}
        selectionMode={SelectionMode.Partial}
        selectionOnDrag={tool === 'box-select'}
        snapGrid={SNAP_GRID}
        snapToGrid={workflowSnapToGrid || isSnapHeld}
        style={{ background: 'transparent' }}
        onConnect={onConnect}
        onConnectEnd={onConnectEnd}
        onEdgeClick={onEdgeClick}
        onEdgesChange={onEdgesChange}
        onInit={setFlowInstance}
        onNodeClick={onNodeClick}
        onNodeContextMenu={onNodeContextMenu}
        onNodeMouseEnter={onNodeMouseEnter}
        onNodeMouseLeave={onNodeMouseLeave}
        onNodesChange={onNodesChange}
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
        {workflowShowMinimap ? <FlowMiniMap /> : null}
      </ReactFlow>
      {lassoOverlay}
      {eraserOverlay}
      <NodeContextMenu
        canPaste={hasClipboardNodes}
        menuState={contextMenu}
        onAddConnector={(position) => {
          addConnector(position);
          setContextMenu(null);
        }}
        onClose={() => setContextMenu(null)}
        onCopy={() => {
          copyNodes(contextMenu?.kind === 'node' ? contextMenu.nodeId : undefined);
          setContextMenu(null);
        }}
        onDelete={() => {
          deleteNodes(contextMenu?.kind === 'node' ? contextMenu.nodeId : undefined);
          setContextMenu(null);
        }}
        onDuplicate={() => {
          duplicateNodes(contextMenu?.kind === 'node' ? contextMenu.nodeId : undefined);
          setContextMenu(null);
        }}
        onPaste={() => {
          pasteNodes();
          setContextMenu(null);
        }}
        onToggleOpen={() => {
          if (contextMenu?.kind === 'node' && contextMenu.nodeId && contextMenu.isNodeOpen !== null) {
            dispatch({
              action: { isOpen: !contextMenu.isNodeOpen, nodeId: contextMenu.nodeId, type: 'setNodeIsOpen' },
              type: 'applyProjectGraphAction',
            });
          }

          setContextMenu(null);
        }}
      />
    </Box>
  );
};

const ReadinessBanner = () => {
  const projectGraph = useActiveProjectSelector((project) => project.projectGraph);
  const templatesSnapshot = useInvocationTemplatesSnapshot();
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
        {readiness.reasons.slice(0, 4).map((reason) => (
          <Text key={reason} color="fg.muted" fontSize="2xs">
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

export const WorkflowEditorView = () => {
  useEffect(() => {
    ensureInvocationTemplatesLoaded();
  }, []);

  return (
    <ReactFlowProvider>
      <Flex direction="column" h="full" minH="0" w="full">
        <Box flex="1" minH="0" position="relative">
          <WorkflowFlow />
          <ReadinessBanner />
        </Box>
      </Flex>
    </ReactFlowProvider>
  );
};
