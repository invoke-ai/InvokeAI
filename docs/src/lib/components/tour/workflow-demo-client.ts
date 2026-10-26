// The browser half of `WorkflowDemo.astro`: node dragging and keyboard moves, collapsing, and rewiring edges. The page
// already holds the initial render; this keeps it in step with the graph state as it changes.
import {
  EDGE_INSET,
  WORLD,
  WORLD_BOUNDS,
  canConnect,
  clampPosition,
  connectEdge,
  createInitialGraph,
  findNearest,
  getConnectedInputs,
  getEdgeId,
  getEdgePath,
  getFieldTypeColor,
  getFieldTypeLabel,
  getHandlePoint,
  layoutNode,
  toGraphBounds,
} from './workflow-demo.mjs';

type Graph = ReturnType<typeof createInitialGraph>;
type GraphNode = Graph['nodes'][number];
type GraphEdge = Graph['edges'][number];
type NodeLayout = ReturnType<typeof layoutNode>;
type Point = { x: number; y: number };

const SVG_NS = 'http://www.w3.org/2000/svg';
const KEY_STEP = 10;
const KEY_STEP_LARGE = 50;
/** Movement in screen pixels before a press on a header becomes a drag. */
const DRAG_THRESHOLD = 3;
/** How near, in graph pixels, a dropped connection snaps to a handle (xyflow's default connection radius). */
const CONNECTION_RADIUS = 20;
const ARROW_DELTAS: Record<string, Point> = {
  ArrowUp: { x: 0, y: -1 },
  ArrowDown: { x: 0, y: 1 },
  ArrowLeft: { x: -1, y: 0 },
  ArrowRight: { x: 1, y: 0 },
};

export const mountWorkflowDemo = (root: HTMLElement) => {
  const stage = root.querySelector<HTMLElement>('[data-stage]');
  const edgeLayer = root.querySelector<SVGGElement>('[data-edges]');
  const connectionPath = root.querySelector<SVGPathElement>('[data-connection]');
  const announcer = root.querySelector<HTMLElement>('[data-announcer]');
  if (!stage || !edgeLayer || !connectionPath || !announcer) return;

  const { nodes, edges: initialEdges } = createInitialGraph();
  let edges: GraphEdge[] = initialEdges;
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const nodeElements = new Map<string, HTMLElement>();
  for (const node of nodes) {
    const element = root.querySelector<HTMLElement>(`[data-node="${node.id}"]`);
    if (!element) return;
    nodeElements.set(node.id, element);
  }
  const layouts = new Map<string, NodeLayout>();
  let selectedId: string | null = null;
  let topZ = nodes.length;

  const handleElement = (nodeId: string, kind: 'input' | 'output', name: string) =>
    nodeElements.get(nodeId)!.querySelector<HTMLElement>(`[data-handle][data-kind="${kind}"][data-name="${name}"]`);

  const edgeElements = () => [...edgeLayer.querySelectorAll<SVGGElement>('.wf-edge')];

  // Nodes may roam the whole element marked `data-workflow-bounds` (the home page's Workflows row, copy included),
  // or just the graph's own box without one.
  const boundsElement = root.closest<HTMLElement>('[data-workflow-bounds]');

  /**
   * Where the scaled graph sits on screen and the area nodes may move in, in graph coordinates. Measured once per press
   * and reused for the whole gesture, so moves only write styles; a resize or scroll invalidates it.
   */
  let geometry: { stage: DOMRect; scale: number; bounds: typeof WORLD_BOUNDS } | null = null;
  const getGeometry = () => {
    const stageRect = geometry ? null : stage.getBoundingClientRect();
    // Not laid out yet (hidden): keep to the graph's own box until a resize measures it again.
    if (stageRect && stageRect.width === 0) return { stage: stageRect, scale: 1, bounds: WORLD_BOUNDS };
    if (stageRect) {
      geometry = {
        stage: stageRect,
        scale: stageRect.width / WORLD.width,
        bounds: boundsElement ? toGraphBounds(boundsElement.getBoundingClientRect(), stageRect) : WORLD_BOUNDS,
      };
    }
    return geometry;
  };

  /** Graph coordinates of a pointer event: the stage is scaled to fit its container. */
  const toGraphPoint = (event: PointerEvent): Point => {
    const { stage: rect, scale } = getGeometry();
    return { x: (event.clientX - rect.left) / scale, y: (event.clientY - rect.top) / scale };
  };

  const placeNode = (node: GraphNode, position: Point) => {
    const layout = layouts.get(node.id)!;
    const size = { width: node.width, height: layout.height };
    const clamped = clampPosition(position, size, getGeometry().bounds, EDGE_INSET);
    node.x = clamped.x;
    node.y = clamped.y;
    nodeElements.get(node.id)!.style.translate = `${node.x}px ${node.y}px`;
  };

  /** Recomputes a node's height and handle positions after its fields or open state change. */
  const relayoutNode = (node: GraphNode) => {
    const connected = getConnectedInputs(edges, node.id);
    const layout = layoutNode(node, connected);
    layouts.set(node.id, layout);
    const element = nodeElements.get(node.id)!;
    for (const row of element.querySelectorAll<HTMLElement>('[data-input]')) {
      row.dataset.connected = String(connected.has(row.dataset.input!));
    }
    for (const [kind, positions] of [
      ['input', layout.inputs],
      ['output', layout.outputs],
    ] as const) {
      for (const [name, y] of Object.entries(positions)) {
        const handle = handleElement(node.id, kind, name);
        if (handle) handle.style.top = `${y}px`;
      }
    }
    // A node that grew past the bottom edge moves up to stay inside its bounds.
    placeNode(node, node);
  };

  const edgePath = (edge: GraphEdge) => {
    const source = nodeById.get(edge.source)!;
    const target = nodeById.get(edge.target)!;
    const from = getHandlePoint(source, layouts.get(source.id)!, 'output', edge.sourceHandle);
    const to = getHandlePoint(target, layouts.get(target.id)!, 'input', edge.targetHandle);
    return from && to ? getEdgePath(from, to) : '';
  };

  const updateEdgePaths = (nodeId?: string) => {
    for (const element of edgeElements()) {
      if (nodeId && element.dataset.source !== nodeId && element.dataset.target !== nodeId) continue;
      const edge = edges.find((candidate) => getEdgeId(candidate) === element.dataset.edge);
      if (!edge) continue;
      const d = edgePath(edge);
      for (const path of element.querySelectorAll('path')) path.setAttribute('d', d);
    }
  };

  const createEdgeElement = (edge: GraphEdge) => {
    const type = nodeById.get(edge.source)!.outputs.find((output) => output.name === edge.sourceHandle)!.type;
    const group = document.createElementNS(SVG_NS, 'g');
    group.classList.add('wf-edge');
    group.dataset.edge = getEdgeId(edge);
    group.dataset.source = edge.source;
    group.dataset.sourceHandle = edge.sourceHandle;
    group.dataset.target = edge.target;
    group.dataset.targetHandle = edge.targetHandle;
    group.style.setProperty('--wf-edge-color', getFieldTypeColor(type));
    const title = document.createElementNS(SVG_NS, 'title');
    title.textContent = getFieldTypeLabel(type);
    const path = document.createElementNS(SVG_NS, 'path');
    path.classList.add('wf-edge__path');
    const hit = document.createElementNS(SVG_NS, 'path');
    hit.classList.add('wf-edge__hit');
    group.append(title, path, hit);
    return group;
  };

  /** Brings the edge elements in line with `edges`, keeping the elements of edges that did not change. */
  const syncEdges = () => {
    const ids = new Set(edges.map(getEdgeId));
    for (const element of edgeElements()) if (!ids.has(element.dataset.edge!)) element.remove();
    const existing = new Set(edgeElements().map((element) => element.dataset.edge));
    for (const edge of edges) if (!existing.has(getEdgeId(edge))) edgeLayer.append(createEdgeElement(edge));
    markSelectedEdges();
    updateEdgePaths();
  };

  const markSelectedEdges = () => {
    for (const element of edgeElements()) {
      const selected = selectedId !== null && (element.dataset.source === selectedId || element.dataset.target === selectedId);
      element.toggleAttribute('data-selected', selected);
    }
  };

  const select = (nodeId: string | null) => {
    if (selectedId === nodeId) return;
    if (selectedId) nodeElements.get(selectedId)!.removeAttribute('data-selected');
    selectedId = nodeId;
    if (nodeId) {
      const element = nodeElements.get(nodeId)!;
      element.setAttribute('data-selected', '');
      // The selected node rises above the others, as in the editor.
      element.style.zIndex = String(++topZ);
    }
    markSelectedEdges();
  };

  const describeHandle = (nodeId: string, kind: 'input' | 'output', name: string) => {
    const node = nodeById.get(nodeId)!;
    const field = (kind === 'input' ? node.inputs : node.outputs).find((candidate) => candidate.name === name);
    return `${field?.label ?? name} of ${node.title}`;
  };

  const announce = (message: string) => {
    announcer.textContent = message;
  };

  for (const node of nodes) relayoutNode(node);
  syncEdges();

  for (const node of nodes) {
    const element = nodeElements.get(node.id)!;
    const header = element.querySelector<HTMLElement>('[data-drag-handle]')!;
    const collapse = element.querySelector<HTMLButtonElement>('[data-collapse]')!;

    element.addEventListener('focusin', () => select(node.id));
    element.addEventListener('focusout', (event) => {
      if (!element.contains(event.relatedTarget as Node | null)) select(null);
    });

    element.addEventListener('keydown', (event) => {
      const delta = ARROW_DELTAS[event.key];
      if (!delta || event.target !== element || event.altKey || event.ctrlKey || event.metaKey) return;
      event.preventDefault();
      const step = event.shiftKey ? KEY_STEP_LARGE : KEY_STEP;
      placeNode(node, { x: node.x + delta.x * step, y: node.y + delta.y * step });
      updateEdgePaths(node.id);
    });

    collapse.addEventListener('click', () => {
      node.open = !node.open;
      element.dataset.open = String(node.open);
      collapse.setAttribute('aria-expanded', String(node.open));
      collapse.setAttribute('aria-label', node.open ? 'Collapse node' : 'Expand node');
      relayoutNode(node);
      updateEdgePaths(node.id);
    });

    let drag: { origin: Point; screen: Point; moved: boolean } | null = null;

    header.addEventListener('pointerdown', (event) => {
      if (event.button !== 0 || (event.target as Element).closest('button')) return;
      event.preventDefault();
      element.focus({ preventScroll: true });
      header.setPointerCapture(event.pointerId);
      geometry = null;
      getGeometry();
      drag = { origin: { x: node.x, y: node.y }, screen: { x: event.clientX, y: event.clientY }, moved: false };
    });

    header.addEventListener('pointermove', (event) => {
      if (!drag || !header.hasPointerCapture(event.pointerId)) return;
      const dx = event.clientX - drag.screen.x;
      const dy = event.clientY - drag.screen.y;
      if (!drag.moved) {
        if (Math.hypot(dx, dy) < DRAG_THRESHOLD) return;
        drag.moved = true;
        element.setAttribute('data-dragging', '');
      }
      const { scale } = getGeometry();
      placeNode(node, { x: drag.origin.x + dx / scale, y: drag.origin.y + dy / scale });
      updateEdgePaths(node.id);
    });

    header.addEventListener('lostpointercapture', () => {
      drag = null;
      element.removeAttribute('data-dragging');
    });
  }

  // Hovering a handle highlights the edges attached to it.
  for (const handle of root.querySelectorAll<HTMLElement>('[data-handle]')) {
    const nodeId = handle.closest<HTMLElement>('[data-node]')!.dataset.node!;
    const attached = () =>
      edgeElements().filter((element) =>
        handle.dataset.kind === 'output'
          ? element.dataset.source === nodeId && element.dataset.sourceHandle === handle.dataset.name
          : element.dataset.target === nodeId && element.dataset.targetHandle === handle.dataset.name,
      );
    handle.addEventListener('pointerenter', () => attached().forEach((element) => element.setAttribute('data-hover', '')));
    handle.addEventListener('pointerleave', () => attached().forEach((element) => element.removeAttribute('data-hover')));
  }

  // Connecting: drag from an output to a matching input. Dragging from a connected input picks its edge up again,
  // to move it to another input or drop it on the canvas to remove it.
  let connection: {
    source: { node: string; handle: string };
    from: Point;
    lifted: GraphEdge | null;
    target: { node: string; handle: string; x: number; y: number } | null;
    candidates: { node: string; handle: string; x: number; y: number; element: HTMLElement }[];
  } | null = null;

  const endConnection = (commit: boolean) => {
    if (!connection) return;
    const { lifted, source, target, candidates } = connection;
    connection = null;
    root.removeAttribute('data-connecting');
    connectionPath.setAttribute('d', '');
    for (const candidate of candidates) candidate.element.removeAttribute('data-valid');
    root.querySelector('[data-snapped]')?.removeAttribute('data-snapped');
    if (lifted) edgeLayer.querySelector(`[data-edge="${getEdgeId(lifted)}"]`)?.removeAttribute('data-pending');
    if (!commit) return;

    const before = edges;
    if (lifted) edges = edges.filter((edge) => edge !== lifted);
    if (target) {
      edges = connectEdge(edges, {
        source: source.node,
        sourceHandle: source.handle,
        target: target.node,
        targetHandle: target.handle,
      });
      announce(
        `Connected ${describeHandle(source.node, 'output', source.handle)} to ${describeHandle(target.node, 'input', target.handle)}.`,
      );
    } else if (lifted) {
      announce(`Removed the connection to ${describeHandle(lifted.target, 'input', lifted.targetHandle)}.`);
    }
    if (edges === before) return;
    const touched = new Set([...before, ...edges].map((edge) => edge.target));
    for (const nodeId of touched) relayoutNode(nodeById.get(nodeId)!);
    syncEdges();
  };

  const drawConnection = (pointer: Point) => {
    if (!connection) return;
    const snapped = findNearest(connection.candidates, pointer, CONNECTION_RADIUS);
    root.querySelector('[data-snapped]')?.removeAttribute('data-snapped');
    snapped?.element.setAttribute('data-snapped', '');
    connection.target = snapped;
    connectionPath.setAttribute('d', getEdgePath(connection.from, snapped ?? pointer));
  };

  for (const handle of root.querySelectorAll<HTMLElement>('[data-handle]')) {
    const nodeId = handle.closest<HTMLElement>('[data-node]')!.dataset.node!;
    const name = handle.dataset.name!;
    const kind = handle.dataset.kind as 'input' | 'output';

    handle.addEventListener('pointerdown', (event) => {
      if (event.button !== 0 || !nodeById.get(nodeId)!.open) return;
      const lifted =
        kind === 'input' ? (edges.find((edge) => edge.target === nodeId && edge.targetHandle === name) ?? null) : null;
      if (kind === 'input' && !lifted) return;
      event.preventDefault();
      event.stopPropagation();
      geometry = null;
      const source = lifted ? { node: lifted.source, handle: lifted.sourceHandle } : { node: nodeId, handle: name };
      const sourceNode = nodeById.get(source.node)!;
      const candidates = nodes
        .filter((node) => node.open)
        .flatMap((node) =>
          node.inputs
            .filter((input) => canConnect(nodes, source, { node: node.id, handle: input.name }))
            .map((input) => {
              const point = getHandlePoint(node, layouts.get(node.id)!, 'input', input.name)!;
              const element = handleElement(node.id, 'input', input.name)!;
              element.setAttribute('data-valid', '');
              return { node: node.id, handle: input.name, ...point, element };
            }),
        );
      connection = {
        source,
        from: getHandlePoint(sourceNode, layouts.get(sourceNode.id)!, 'output', source.handle)!,
        lifted,
        target: null,
        candidates,
      };
      if (lifted) edgeLayer.querySelector(`[data-edge="${getEdgeId(lifted)}"]`)?.setAttribute('data-pending', '');
      root.setAttribute('data-connecting', '');
      handle.setPointerCapture(event.pointerId);
      drawConnection(toGraphPoint(event));
    });

    handle.addEventListener('pointermove', (event) => {
      if (connection && handle.hasPointerCapture(event.pointerId)) drawConnection(toGraphPoint(event));
    });
    handle.addEventListener('pointerup', () => endConnection(true));
    handle.addEventListener('pointercancel', () => endConnection(false));
    handle.addEventListener('lostpointercapture', () => endConnection(false));
  }

  // A resize changes the graph's scale and the area around it: measure again and pull nodes back inside.
  const invalidate = () => {
    geometry = null;
  };
  addEventListener('scroll', invalidate, { capture: true, passive: true });
  const resizeObserver = new ResizeObserver(() => {
    invalidate();
    for (const node of nodes) placeNode(node, node);
    updateEdgePaths();
  });
  resizeObserver.observe(root);
  if (boundsElement) resizeObserver.observe(boundsElement);

  root.setAttribute('data-ready', '');
};
