// @ts-check
// The home page's interactive workflow editor demo: its graph, and the geometry both the build-time render and the
// browser script draw it with. Row heights are fixed (the component's CSS uses the same numbers), so handle positions
// and edge paths are computed here instead of measured, and the page renders identically without JavaScript.
// Sizes, colors and the edge curve follow the app's node editor:
// `invokeai/frontend/webv2/src/features/workflow/ui/{nodeChrome.tsx,editor/InvocationFlowNode.tsx,editor/WorkflowEdge.tsx}`.

/**
 * @typedef {'switch' | 'number' | 'select' | 'image'} WidgetKind
 * `connectionOnly` inputs take no value of their own (the app's `input: 'connection'`), so they cannot be pinned to the
 * workflow's form; `handle: false` inputs take no connection.
 * @typedef {{ name: string, label: string, type: string, required?: boolean, connectionOnly?: boolean, handle?: boolean,
 *   widget?: WidgetKind }} InputField
 * @typedef {{ name: string, label: string, type: string }} OutputField
 * @typedef {{ id: string, title: string, x: number, y: number, width: number, open: boolean, footer: boolean,
 *   outputs: OutputField[], inputs: InputField[] }} GraphNode
 * @typedef {{ source: string, sourceHandle: string, target: string, targetHandle: string }} GraphEdge
 * @typedef {{ x: number, y: number }} Point
 * @typedef {{ height: number, inputs: Record<string, number>, outputs: Record<string, number> }} NodeLayout
 * @typedef {{ left: number, top: number, right: number, bottom: number }} Bounds
 */

/** The graph's coordinate space. The component scales it down to fit narrower containers. */
export const WORLD = { width: 680, height: 440 };

/** The graph's own box, for clamping when no larger area is given. */
export const WORLD_BOUNDS = { left: 0, top: 0, right: WORLD.width, bottom: WORLD.height };

/** How far nodes keep inside their bounds: the handles on a node's sides reach 6px past it. */
export const EDGE_INSET = 6;

/** Pixel sizes of the node chrome, shared with `WorkflowDemo.astro`'s CSS. */
export const SIZES = {
  border: 1,
  // A 24px icon button (`size="sm"`) with 6px of padding above and below.
  header: 36,
  bodyPad: 4,
  // A 20px label line with 2px of padding above and below.
  row: 24,
  widgetGap: 2,
  footer: 32,
  // Collapsed nodes keep their handles mounted 14px above the header's bottom edge, as `HiddenHandles` does.
  collapsedHandleOffset: 14,
};

/** Control heights below the label row: md inputs and selects are 28px, the image well is 90px plus a 24px button row. */
export const WIDGET_HEIGHTS = { switch: 18, number: 28, select: 28, image: 118 };

/** The app's handle and edge colors (`features/workflow/core/fields.ts`), for the field types this graph uses. */
export const FIELD_TYPE_COLORS = {
  BooleanField: '#4ade80',
  ImageField: '#c4b5fd',
  IntegerField: '#f87171',
  LatentsField: '#f9a8d4',
  VAEField: '#2563eb',
};

/** Model field types get square handles. */
const ANGULAR_FIELD_TYPES = new Set(['VAEField']);

/** @param {string} type */
export const getFieldTypeColor = (type) => FIELD_TYPE_COLORS[/** @type {keyof typeof FIELD_TYPE_COLORS} */ (type)] ?? '#9ca3af';

/** @param {string} type */
export const isAngularFieldType = (type) => ANGULAR_FIELD_TYPES.has(type);

/** `ImageField` reads "Image", as the app's handle tooltips and edge titles word it. */
export const getFieldTypeLabel = (/** @type {string} */ type) => (type === 'VAEField' ? 'VAE' : type.replace(/Field$/, ''));

/** The graph the demo starts with: a decoded image composited over a background, with its threshold from a primitive. */
export const createInitialGraph = () => {
  /** @type {GraphNode[]} */
  const nodes = [
    {
      id: 'l2i',
      title: 'Latents to Image - SD1.5, SDXL',
      x: 10,
      y: 10,
      width: 256,
      open: true,
      footer: true,
      outputs: [
        { name: 'image', label: 'Image', type: 'ImageField' },
        { name: 'width', label: 'Width', type: 'IntegerField' },
        { name: 'height', label: 'Height', type: 'IntegerField' },
      ],
      inputs: [
        { name: 'latents', label: 'Latents', type: 'LatentsField', required: true, connectionOnly: true },
        { name: 'vae', label: 'Vae', type: 'VAEField', required: true, connectionOnly: true },
        { name: 'tiled', label: 'Tiled', type: 'BooleanField', widget: 'switch' },
      ],
    },
    {
      id: 'integer',
      title: 'Integer Primitive',
      x: 40,
      y: 294,
      width: 180,
      open: true,
      footer: false,
      outputs: [{ name: 'value', label: 'Value', type: 'IntegerField' }],
      inputs: [{ name: 'value', label: 'Value', type: 'IntegerField', widget: 'number' }],
    },
    {
      id: 'compositor',
      title: 'Image Compositor',
      x: 422,
      y: 8,
      width: 248,
      open: true,
      footer: true,
      outputs: [
        { name: 'image', label: 'Image', type: 'ImageField' },
        { name: 'width', label: 'Width', type: 'IntegerField' },
        { name: 'height', label: 'Height', type: 'IntegerField' },
      ],
      inputs: [
        { name: 'board', label: 'Board', type: 'BoardField', handle: false, widget: 'select' },
        { name: 'subject', label: 'Image Subject', type: 'ImageField', required: true },
        { name: 'background', label: 'Image Background', type: 'ImageField', required: true, widget: 'image' },
        { name: 'threshold', label: 'Threshold', type: 'IntegerField', widget: 'number' },
      ],
    },
  ];
  /** @type {GraphEdge[]} */
  const edges = [
    { source: 'l2i', sourceHandle: 'image', target: 'compositor', targetHandle: 'subject' },
    { source: 'integer', sourceHandle: 'value', target: 'compositor', targetHandle: 'threshold' },
  ];
  return { nodes, edges };
};

/** @param {GraphEdge} edge */
export const getEdgeId = (edge) => `${edge.source}.${edge.sourceHandle}-${edge.target}.${edge.targetHandle}`;

/** The input names of a node that an edge feeds. @param {GraphEdge[]} edges @param {string} nodeId */
export const getConnectedInputs = (edges, nodeId) =>
  new Set(edges.filter((edge) => edge.target === nodeId).map((edge) => edge.targetHandle));

/** A connected input shows no control: its value comes from the edge. @param {InputField} input @param {boolean} connected */
export const getInputRowHeight = (input, connected) =>
  SIZES.row + (input.widget && !connected ? SIZES.widgetGap + WIDGET_HEIGHTS[input.widget] : 0);

/**
 * The node's height and its handles' vertical centers, measured from the node's top edge. Outputs list first, then
 * inputs, as in the app. A collapsed node is its header alone, with every handle stacked behind it.
 * @param {GraphNode} node
 * @param {Set<string>} connectedInputs
 * @returns {NodeLayout}
 */
export const layoutNode = (node, connectedInputs) => {
  /** @type {NodeLayout} */
  const layout = { height: 0, inputs: {}, outputs: {} };
  if (!node.open) {
    const y = SIZES.border + SIZES.header - SIZES.collapsedHandleOffset;
    for (const output of node.outputs) layout.outputs[output.name] = y;
    for (const input of node.inputs) if (input.handle !== false) layout.inputs[input.name] = y;
    layout.height = SIZES.border * 2 + SIZES.header;
    return layout;
  }

  let top = SIZES.border + SIZES.header + SIZES.bodyPad;
  const center = SIZES.row / 2;
  for (const output of node.outputs) {
    layout.outputs[output.name] = top + center;
    top += SIZES.row;
  }
  for (const input of node.inputs) {
    if (input.handle !== false) layout.inputs[input.name] = top + center;
    top += getInputRowHeight(input, connectedInputs.has(input.name));
  }
  layout.height = top + SIZES.bodyPad + (node.footer ? SIZES.footer : 0) + SIZES.border;
  return layout;
};

/**
 * A handle's center in graph coordinates: inputs sit on the node's left edge, outputs on its right.
 * @param {GraphNode} node @param {NodeLayout} layout @param {'input' | 'output'} kind @param {string} name
 * @returns {Point | null}
 */
export const getHandlePoint = (node, layout, kind, name) => {
  const y = (kind === 'input' ? layout.inputs : layout.outputs)[name];
  if (y === undefined) return null;
  return { x: kind === 'input' ? node.x : node.x + node.width, y: node.y + y };
};

/** xyflow's control-point offset for a bezier edge (`@xyflow/system`, curvature 0.25). @param {number} distance */
const controlOffset = (distance) => (distance >= 0 ? 0.5 * distance : 0.25 * 25 * Math.sqrt(-distance));

const round = (/** @type {number} */ value) => Math.round(value * 100) / 100;

/**
 * The SVG path of a bezier edge leaving a right-hand source handle and entering a left-hand target handle, the
 * curve xyflow's `getBezierPath` draws for the app's default edges. A target behind its source loops out and back.
 * @param {Point} source @param {Point} target
 */
export const getEdgePath = (source, target) => {
  const sourceControlX = source.x + controlOffset(target.x - source.x);
  const targetControlX = target.x - controlOffset(target.x - source.x);
  return (
    `M${round(source.x)},${round(source.y)} ` +
    `C${round(sourceControlX)},${round(source.y)} ${round(targetControlX)},${round(target.y)} ${round(target.x)},${round(target.y)}`
  );
};

/**
 * Keeps a node of the given size inside the bounds, `inset` pixels in from each edge. A node taller or wider than that
 * space pins to the top or left, so its header stays reachable.
 * @param {Point} position @param {{ width: number, height: number }} size @param {Bounds} bounds @param {number} [inset]
 * @returns {Point}
 */
export const clampPosition = (position, size, bounds, inset = 0) => ({
  x: Math.round(Math.max(bounds.left + inset, Math.min(position.x, bounds.right - inset - size.width))),
  y: Math.round(Math.max(bounds.top + inset, Math.min(position.y, bounds.bottom - inset - size.height))),
});

/**
 * An on-screen area in graph coordinates. The graph is drawn at `WORLD` size and scaled to its box on the page, so an
 * element around it (the whole tour section) maps to a larger area that may start at negative coordinates.
 * @param {Bounds} area the area's client rect @param {Bounds & { width: number }} stage the graph's client rect
 * @returns {Bounds}
 */
export const toGraphBounds = (area, stage) => {
  const scale = stage.width / WORLD.width;
  return {
    left: (area.left - stage.left) / scale,
    top: (area.top - stage.top) / scale,
    right: (area.right - stage.left) / scale,
    bottom: (area.bottom - stage.top) / scale,
  };
};

/**
 * Whether an output may feed an input: matching field types, between two different nodes.
 * @param {GraphNode[]} nodes
 * @param {{ node: string, handle: string }} source an output
 * @param {{ node: string, handle: string }} target an input
 */
export const canConnect = (nodes, source, target) => {
  if (source.node === target.node) return false;
  const output = nodes.find((node) => node.id === source.node)?.outputs.find((field) => field.name === source.handle);
  const input = nodes.find((node) => node.id === target.node)?.inputs.find((field) => field.name === target.handle);
  return Boolean(output && input && input.handle !== false && output.type === input.type);
};

/**
 * Adds an edge. An input takes one connection, so an edge already feeding the same input is replaced.
 * @param {GraphEdge[]} edges @param {GraphEdge} edge
 * @returns {GraphEdge[]}
 */
export const connectEdge = (edges, edge) => [
  ...edges.filter((existing) => !(existing.target === edge.target && existing.targetHandle === edge.targetHandle)),
  edge,
];

/**
 * The candidate nearest the pointer within `radius`, like xyflow's connection radius: a drop near a handle snaps to it.
 * @template {Point} T
 * @param {T[]} candidates @param {Point} pointer @param {number} radius
 * @returns {T | null}
 */
export const findNearest = (candidates, pointer, radius) => {
  let best = null;
  let bestDistance = radius;
  for (const candidate of candidates) {
    const distance = Math.hypot(candidate.x - pointer.x, candidate.y - pointer.y);
    if (distance <= bestDistance) {
      best = candidate;
      bestDistance = distance;
    }
  }
  return best;
};
