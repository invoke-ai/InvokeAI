import type { PreviewGraphLike } from './graphToDocument';
import type { WorkflowModelRequirement } from './modelRequirements';
import type {
  FieldInputTemplate,
  InvocationTemplate,
  InvocationTemplates,
  ProjectGraphState,
  WorkflowFormElement,
  WorkflowInvocationNode,
  XYPosition,
} from './types';

import { isModelFieldType } from './fields';
import { getCanonicalWorkflowEdges } from './forLoops';
import { getNodeDepths, getTopologicalOrder } from './graphLayout';
import { extractNodeModelRequirements } from './modelRequirements';
import { isInvocationNode } from './types';

/**
 * The library's read-only preview of a saved workflow. Drawn from the document, not the compiled graph, so a workflow
 * that does not compile still previews.
 */

export type WorkflowPreviewValue =
  | { kind: 'text'; text: string }
  | { kind: 'number'; value: number }
  | { kind: 'boolean'; value: boolean }
  | { kind: 'model'; name: string }
  | { kind: 'image'; name: string }
  | { kind: 'list'; count: number };

export interface WorkflowPreviewField {
  name: string;
  /** The author's label, else the template's title, else the field name. */
  label: string;
  value: WorkflowPreviewValue;
  isExposed: boolean;
}

export interface WorkflowPreviewNode {
  id: string;
  type: string;
  /** The node's own label, else its template's title, else its type id. */
  title: string;
  /** No template on this server: its node pack is missing. */
  isUnknown: boolean;
  notes: string;
  /** Inputs with a value of their own (not fed by a connection), in template order. */
  fields: readonly WorkflowPreviewField[];
  values: Readonly<Record<string, unknown>>;
  /** Its first exposed field with a value short enough for a one-line summary. */
  exposedSummary: WorkflowPreviewField | null;
  isExposed: boolean;
  /** Nothing consumes it and it is not a helper: what the workflow makes. */
  isResult: boolean;
  modelRequirements: readonly WorkflowModelRequirement[];
}

export interface WorkflowPreviewEdge {
  id: string;
  source: string;
  target: string;
  isRerouted: boolean;
  isLoopLinkage: boolean;
}

/** A folded value-only helper, drawn on the node it feeds. */
export interface WorkflowPreviewPill {
  nodeId: string;
  /** The form's label for its value when exposed, else the helper's title. */
  label: string;
  value: WorkflowPreviewValue | null;
}

export interface WorkflowPreviewLayout {
  /** Every chip's width in this layout: one per drawing, so columns stay aligned. */
  nodeWidth: number;
  /** Drawn nodes in topological order, which is also their keyboard order. */
  nodeIds: readonly string[];
  edges: readonly WorkflowPreviewEdge[];
  /** Each chip's top-left corner; a chip's pills stack above it, `pillHeight` each. */
  positions: Readonly<Record<string, XYPosition>>;
  /** Folded helpers per consumer; empty in the full view. */
  pills: Readonly<Record<string, readonly WorkflowPreviewPill[]>>;
  /** Each drawn node's drawn neighbours, each once, in keyboard order. */
  neighbours: Readonly<Record<string, { sources: readonly string[]; targets: readonly string[] }>>;
}

export interface WorkflowPreviewInput {
  id: string;
  nodeId: string;
  fieldName: string;
  label: string;
  nodeTitle: string;
  /** The saved value; null when empty. */
  value: WorkflowPreviewValue | null;
  isModelField: boolean;
  modelRequirements: readonly WorkflowModelRequirement[];
}

export type WorkflowPreviewFormItem =
  | { kind: 'heading'; id: string; text: string }
  | { kind: 'text'; id: string; text: string }
  | { kind: 'divider'; id: string }
  | { kind: 'container'; id: string; layout: 'row' | 'column'; children: readonly WorkflowPreviewFormItem[] }
  | { kind: 'input'; id: string; input: WorkflowPreviewInput };

/** The form's inputs under one heading (null before the first heading), as the rail summarises them. */
export interface WorkflowPreviewInputGroup {
  heading: string | null;
  labels: readonly string[];
}

export interface WorkflowPreviewModel {
  nodes: ReadonlyMap<string, WorkflowPreviewNode>;
  /** Node ids and connections only: equal across value-only changes, so a recompiled live source keeps its drawing. */
  structureKey: string;
  /** Every node at the author's positions, scaled to the chips. */
  all: WorkflowPreviewLayout;
  /** Helpers folded and plumbing hidden, in a layered layout; null when nothing folds or hides. */
  simplified: WorkflowPreviewLayout | null;
  form: readonly WorkflowPreviewFormItem[];
  inputs: readonly WorkflowPreviewInput[];
  inputGroups: readonly WorkflowPreviewInputGroup[];
}

export interface WorkflowPreviewMetrics {
  nodeWidth: number;
  nodeHeight: number;
  pillHeight: number;
  layerGap: number;
  rowGap: number;
  /** The author's positions are laid out for editor-sized nodes; the full view scales them by this. */
  authorScale: number;
}

/**
 * Layers stay close, so a workflow six layers deep fits a ~870px stage at a readable zoom; rows sit well apart, since a
 * layered graph is wide and short and the stage has the height to spare.
 */
export const WORKFLOW_PREVIEW_METRICS: WorkflowPreviewMetrics = {
  authorScale: 0.5,
  layerGap: 28,
  nodeHeight: 48,
  nodeWidth: 136,
  pillHeight: 26,
  rowGap: 56,
};

export const WORKFLOW_PREVIEW_MIN_CHIP_WIDTH = WORKFLOW_PREVIEW_METRICS.nodeWidth;
export const WORKFLOW_PREVIEW_MAX_CHIP_WIDTH = 200;
const CHIP_WIDTH_STEP = 16;

const metricsByWidth = new Map<number, WorkflowPreviewMetrics>([
  [WORKFLOW_PREVIEW_METRICS.nodeWidth, WORKFLOW_PREVIEW_METRICS],
]);

/** One object per width, since models cache by metrics; the author's arrangement scales with the chips. */
export const getWorkflowPreviewMetrics = (width: number): WorkflowPreviewMetrics => {
  let metrics = metricsByWidth.get(width);

  if (!metrics) {
    metrics = {
      ...WORKFLOW_PREVIEW_METRICS,
      authorScale: (WORKFLOW_PREVIEW_METRICS.authorScale * width) / WORKFLOW_PREVIEW_METRICS.nodeWidth,
      nodeWidth: width,
    };
    metricsByWidth.set(width, metrics);
  }

  return metrics;
};

/** The widest width from `wanted` down, within the bounds, at which the drawing still `fits`; else the minimum. */
export const chooseWorkflowPreviewChipWidth = (wanted: number, fits: (width: number) => boolean): number => {
  const widest = Math.min(
    WORKFLOW_PREVIEW_MAX_CHIP_WIDTH,
    Math.max(WORKFLOW_PREVIEW_MIN_CHIP_WIDTH, Math.ceil(wanted))
  );

  for (let width = widest; width > WORKFLOW_PREVIEW_MIN_CHIP_WIDTH; width -= CHIP_WIDTH_STEP) {
    if (fits(width)) {
      return width;
    }
  }

  return WORKFLOW_PREVIEW_MIN_CHIP_WIDTH;
};

const PLUMBING_TYPES = new Set(['collect', 'core_metadata']);
/** Recorders (metadata) receive almost every parameter, so nothing reroutes through them; a collect does. */
const RECORDING_TYPES = new Set(['core_metadata']);
const VALUE_PRIMITIVE_TYPES = new Set(['string', 'integer', 'float', 'boolean']);
const SCALAR_FIELD_TYPES = new Set(['StringField', 'IntegerField', 'FloatField', 'BooleanField']);
const SHORT_TEXT_LENGTH = 24;

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;

/** Equal plain data (the model's own nodes, values and layouts), so a rebuild can keep what did not change. */
const isSameData = (a: unknown, b: unknown): boolean => {
  if (Object.is(a, b)) {
    return true;
  }

  if (Array.isArray(a)) {
    return Array.isArray(b) && a.length === b.length && a.every((item, index) => isSameData(item, b[index]));
  }

  if (!isRecord(a) || !isRecord(b) || Array.isArray(b)) {
    return false;
  }

  const keys = Object.keys(a);

  return (
    keys.length === Object.keys(b).length && keys.every((key) => Object.hasOwn(b, key) && isSameData(a[key], b[key]))
  );
};

/** `next`, or `previous` where they hold the same data: unchanged parts keep their identity across rebuilds. */
const keepSame = <T>(previous: T | undefined, next: T): T =>
  previous !== undefined && isSameData(previous, next) ? previous : next;

const describeValue = (value: unknown, template: FieldInputTemplate | undefined): WorkflowPreviewValue | null => {
  if (typeof value === 'string') {
    return value.trim() === '' ? null : { kind: 'text', text: template?.uiChoiceLabels?.[value] ?? value };
  }

  if (typeof value === 'number') {
    return Number.isFinite(value) ? { kind: 'number', value } : null;
  }

  if (typeof value === 'boolean') {
    return { kind: 'boolean', value };
  }

  if (Array.isArray(value)) {
    return value.length === 0 ? null : { kind: 'list', count: value.length };
  }

  if (isRecord(value)) {
    if (typeof value.image_name === 'string') {
      return { kind: 'image', name: value.image_name };
    }

    if (typeof value.key === 'string' && value.key.trim() !== '') {
      return { kind: 'model', name: typeof value.name === 'string' && value.name ? value.name : value.key };
    }
  }

  return null;
};

const isShortValue = (value: WorkflowPreviewValue): boolean =>
  value.kind === 'number' ||
  value.kind === 'boolean' ||
  (value.kind === 'text' && value.text.length <= SHORT_TEXT_LENGTH);

/** The inputs a person sets on the node: internal and hidden ones are the backend's. */
const getAuthoredInputs = (template: InvocationTemplate): FieldInputTemplate[] =>
  Object.values(template.inputs).filter((input) => input.fieldKind === 'input' && !input.uiHidden);

const getFieldTemplate = (node: WorkflowInvocationNode, template: InvocationTemplate | undefined, name: string) =>
  (template && Object.hasOwn(template.inputs, name) ? template.inputs[name] : undefined) ??
  (node.data.dynamicInputTemplates && Object.hasOwn(node.data.dynamicInputTemplates, name)
    ? node.data.dynamicInputTemplates[name]
    : undefined);

const getTemplate = (templates: InvocationTemplates, type: string) =>
  Object.hasOwn(templates, type) ? templates[type] : undefined;

const getFieldLabel = (node: WorkflowInvocationNode, template: InvocationTemplate | undefined, name: string) =>
  node.data.inputs[name]?.label || getFieldTemplate(node, template, name)?.title || name;

/** Template order (`uiOrder`, then declaration), then any instance the template no longer lists. */
const getOrderedFieldNames = (node: WorkflowInvocationNode, template: InvocationTemplate | undefined): string[] => {
  const declared = Object.values({ ...template?.inputs, ...node.data.dynamicInputTemplates })
    .map((input, index) => ({ index, input }))
    .sort((a, b) => (a.input.uiOrder ?? Infinity) - (b.input.uiOrder ?? Infinity) || a.index - b.index)
    .map(({ input }) => input.name);
  const known = new Set(declared);

  return [...declared, ...Object.keys(node.data.inputs).filter((name) => !known.has(name))];
};

const fieldKey = (nodeId: string, fieldName: string) => `${nodeId}\u0000${fieldName}`;

const pushUnique = (lists: Map<string, string[]>, key: string, value: string) => {
  const list = lists.get(key);

  if (!list) {
    lists.set(key, [value]);
  } else if (!list.includes(value)) {
    list.push(value);
  }
};

const getNeighbours = (
  nodeIds: readonly string[],
  edges: readonly WorkflowPreviewEdge[]
): WorkflowPreviewLayout['neighbours'] => {
  const rank = new Map(nodeIds.map((id, index) => [id, index]));
  const sources = new Map<string, string[]>();
  const targets = new Map<string, string[]>();

  for (const edge of edges) {
    pushUnique(sources, edge.target, edge.source);
    pushUnique(targets, edge.source, edge.target);
  }

  const ordered = (list: string[] | undefined) =>
    [...(list ?? [])].sort((a, b) => (rank.get(a) ?? 0) - (rank.get(b) ?? 0));

  return Object.fromEntries(
    nodeIds.map((id) => [id, { sources: ordered(sources.get(id)), targets: ordered(targets.get(id)) }])
  );
};

/**
 * Layered layout, one column per depth. Each chip sits level with the middle of its parents, then a column is pushed
 * apart to clear the pills stacked above each chip. Unconnected first-column nodes stack; the topmost pill starts at 0.
 */
const getPreviewLayeredPositions = (
  nodes: readonly { id: string; pillsHeight: number }[],
  edges: readonly { sourceNodeId: string; targetNodeId: string }[],
  metrics: WorkflowPreviewMetrics
): Record<string, XYPosition> => {
  const depths = getNodeDepths([...nodes], [...edges]);
  const pillsHeights = new Map(nodes.map((node) => [node.id, node.pillsHeight]));
  const parents = new Map<string, string[]>();

  for (const edge of edges) {
    pushUnique(parents, edge.targetNodeId, edge.sourceNodeId);
  }

  const layers = new Map<number, string[]>();

  for (const node of nodes) {
    const depth = depths.get(node.id) ?? 0;

    layers.set(depth, [...(layers.get(depth) ?? []), node.id]);
  }

  const positions: Record<string, XYPosition> = {};

  for (const depth of [...layers.keys()].sort((a, b) => a - b)) {
    const x = depth * (metrics.nodeWidth + metrics.layerGap);
    // Every chip is as tall as the others, so the mean of the parents' chip tops levels the chip with their middle.
    const wanted = layers.get(depth)!.map((id, index) => {
      const placed = (parents.get(id) ?? []).filter((parent) => positions[parent] !== undefined);
      const y =
        placed.length === 0 ? null : placed.reduce((sum, parent) => sum + positions[parent]!.y, 0) / placed.length;

      return { id, index, y };
    });
    let bottom = -Infinity;

    for (const { id, y } of wanted.sort((a, b) => (a.y ?? -Infinity) - (b.y ?? -Infinity) || a.index - b.index)) {
      const top = Math.max(y ?? 0, bottom + metrics.rowGap + pillsHeights.get(id)!);

      positions[id] = { x, y: top };
      bottom = top + metrics.nodeHeight;
    }
  }

  const minY = Math.min(...Object.entries(positions).map(([id, position]) => position.y - pillsHeights.get(id)!));

  for (const position of Object.values(positions)) {
    position.y -= minY;
  }

  return positions;
};

/** Skips elements naming a missing node or element and enters each container once, so a malformed form cannot loop. */
const readForm = (
  document: ProjectGraphState,
  toInput: (id: string, nodeId: string, fieldName: string) => WorkflowPreviewInput | null
): WorkflowPreviewFormItem[] => {
  const { elements, rootElementId } = document.form;
  const entered = new Set<string>();
  const read = (element: WorkflowFormElement | undefined): WorkflowPreviewFormItem | null => {
    switch (element?.type) {
      case 'container':
      case 'collapsible': {
        if (entered.has(element.id)) {
          return null;
        }

        entered.add(element.id);
        const children = element.data.children.map((id) => read(elements[id])).filter((item) => item !== null);

        return {
          // The library preview is read-only: show each section expanded, retaining its title and inputs.
          children:
            element.type === 'collapsible'
              ? [{ id: `${element.id}:heading`, kind: 'heading', text: element.data.content }, ...children]
              : children,
          id: element.id,
          kind: 'container',
          layout: element.type === 'container' ? element.data.layout : 'column',
        };
      }
      case 'node-field': {
        const input = toInput(element.id, element.data.fieldIdentifier.nodeId, element.data.fieldIdentifier.fieldName);

        return input ? { id: element.id, input, kind: 'input' } : null;
      }
      case 'heading':
        return { id: element.id, kind: 'heading', text: element.data.content };
      case 'text':
        return { id: element.id, kind: 'text', text: element.data.content };
      case 'divider':
        return { id: element.id, kind: 'divider' };
      default:
        return null;
    }
  };
  const root = read(elements[rootElementId]);

  return root?.kind === 'container' ? [...root.children] : [];
};

const collectInputs = (items: readonly WorkflowPreviewFormItem[], into: WorkflowPreviewInput[] = []) => {
  for (const item of items) {
    if (item.kind === 'input') {
      into.push(item.input);
    } else if (item.kind === 'container') {
      collectInputs(item.children, into);
    }
  }

  return into;
};

/** The form's inputs grouped under the heading before them, in order; groups without inputs drop out. */
const groupInputs = (items: readonly WorkflowPreviewFormItem[]): WorkflowPreviewInputGroup[] => {
  const groups: { heading: string | null; labels: string[] }[] = [{ heading: null, labels: [] }];
  const walk = (list: readonly WorkflowPreviewFormItem[]) => {
    for (const item of list) {
      if (item.kind === 'heading') {
        groups.push({ heading: item.text, labels: [] });
      } else if (item.kind === 'input') {
        groups.at(-1)!.labels.push(item.input.label);
      } else if (item.kind === 'container') {
        walk(item.children);
      }
    }
  };

  walk(items);

  return groups.filter((group) => group.labels.length > 0);
};

const buildModel = (
  document: ProjectGraphState,
  templates: InvocationTemplates,
  metrics: WorkflowPreviewMetrics,
  previous: WorkflowPreviewModel | null
): WorkflowPreviewModel => {
  const invocationNodes = document.nodes.filter(isInvocationNode);
  const nodeIds = new Set(invocationNodes.map((node) => node.id));
  const canonicalEdges = getCanonicalWorkflowEdges(document).filter(
    (edge) => nodeIds.has(edge.source.node_id) && nodeIds.has(edge.destination.node_id)
  );
  const connectedInputs = new Set(
    canonicalEdges.map((edge) => fieldKey(edge.destination.node_id, edge.destination.field))
  );
  const exposedFields = new Set<string>();

  for (const element of Object.values(document.form.elements)) {
    if (element.type === 'node-field' && nodeIds.has(element.data.fieldIdentifier.nodeId)) {
      exposedFields.add(fieldKey(element.data.fieldIdentifier.nodeId, element.data.fieldIdentifier.fieldName));
    }
  }

  const exposedNodes = new Set([...exposedFields].map((key) => key.slice(0, key.indexOf('\u0000'))));
  const requirements = extractNodeModelRequirements(document, templates);
  const incoming = new Map<string, string[]>();
  const outgoing = new Map<string, string[]>();

  for (const edge of canonicalEdges) {
    pushUnique(outgoing, edge.source.node_id, edge.destination.node_id);
    pushUnique(incoming, edge.destination.node_id, edge.source.node_id);
  }

  const layoutEdges = canonicalEdges
    .filter((edge) => edge.type !== 'loop_linkage')
    .map((edge) => ({ sourceNodeId: edge.source.node_id, targetNodeId: edge.destination.node_id }));
  const order = getTopologicalOrder(invocationNodes, layoutEdges);
  const rank = new Map(order.map((id, index) => [id, index]));
  const byRank = (a: string, b: string) => (rank.get(a) ?? 0) - (rank.get(b) ?? 0);
  const documentNodes = new Map(invocationNodes.map((node) => [node.id, node]));

  const isPlumbing = (id: string) => PLUMBING_TYPES.has(documentNodes.get(id)!.data.type);
  /** A value-only primitive's one input: a scalar it holds itself, with nothing feeding the node and something fed. */
  const getFoldableInput = (id: string): FieldInputTemplate | null => {
    const template = getTemplate(templates, documentNodes.get(id)!.data.type);
    const inputs = template ? getAuthoredInputs(template) : [];
    const input = inputs[0];

    return VALUE_PRIMITIVE_TYPES.has(documentNodes.get(id)!.data.type) &&
      inputs.length === 1 &&
      input!.type.cardinality === 'SINGLE' &&
      SCALAR_FIELD_TYPES.has(input!.type.name) &&
      !incoming.has(id) &&
      outgoing.has(id)
      ? input!
      : null;
  };

  const nodes = new Map<string, WorkflowPreviewNode>();

  for (const id of order) {
    const node = documentNodes.get(id)!;
    const template = getTemplate(templates, node.data.type);
    const values: Record<string, unknown> = {};
    const fields: WorkflowPreviewField[] = [];

    for (const name of getOrderedFieldNames(node, template)) {
      const instance = node.data.inputs[name];
      const fieldTemplate = getFieldTemplate(node, template, name);

      if (!instance || connectedInputs.has(fieldKey(id, name)) || fieldTemplate?.uiHidden) {
        continue;
      }

      const value = describeValue(instance.value, fieldTemplate);

      if (value) {
        values[name] = instance.value;
        fields.push({
          isExposed: exposedFields.has(fieldKey(id, name)),
          label: getFieldLabel(node, template, name),
          name,
          value,
        });
      }
    }

    nodes.set(
      id,
      keepSame(previous?.nodes.get(id), {
        exposedSummary: fields.find((field) => field.isExposed && isShortValue(field.value)) ?? null,
        fields,
        id,
        isExposed: exposedNodes.has(id),
        isResult: !outgoing.has(id) && !isPlumbing(id) && getFoldableInput(id) === null,
        isUnknown: template === undefined,
        modelRequirements: (requirements.get(id) ?? []).map(({ requirement }) => requirement),
        notes: node.data.notes,
        title: node.data.label || template?.title || node.data.type,
        type: node.data.type,
        values,
      })
    );
  }

  const form = readForm(document, (id, nodeId, fieldName) => {
    const node = documentNodes.get(nodeId);

    if (!node) {
      return null;
    }

    const template = getTemplate(templates, node.data.type);
    const fieldTemplate = getFieldTemplate(node, template, fieldName);

    return {
      fieldName,
      id,
      isModelField: fieldTemplate
        ? isModelFieldType(fieldTemplate.type) || fieldTemplate.type.name === 'LoRAField'
        : false,
      label: getFieldLabel(node, template, fieldName),
      modelRequirements: (requirements.get(nodeId) ?? [])
        .filter((entry) => entry.fieldName === fieldName)
        .map(({ requirement }) => requirement),
      nodeId,
      nodeTitle: nodes.get(nodeId)!.title,
      value: describeValue(node.data.inputs[fieldName]?.value, fieldTemplate),
    };
  });

  // One drawn edge per connected pair and kind: several fields between two nodes read as one connection.
  const allEdges: WorkflowPreviewEdge[] = [];
  const seenPairs = new Set<string>();

  for (const edge of canonicalEdges) {
    const isLoopLinkage = edge.type === 'loop_linkage';
    const pair = `${edge.source.node_id}\u0000${edge.destination.node_id}\u0000${isLoopLinkage}`;

    if (!seenPairs.has(pair)) {
      seenPairs.add(pair);
      allEdges.push({
        id: edge.id,
        isLoopLinkage,
        isRerouted: false,
        source: edge.source.node_id,
        target: edge.destination.node_id,
      });
    }
  }

  const structureKey = JSON.stringify([
    order.map((id) => `${id}\u0000${documentNodes.get(id)!.data.type}`).sort(),
    allEdges.map((edge) => `${edge.source}\u0000${edge.target}\u0000${edge.isLoopLinkage}`).sort(),
  ]);
  // A rebuild of the same structure (a live source recompiled with new values) keeps the previous layouts.
  const same = previous?.structureKey === structureKey ? previous : null;

  const authorX = Math.min(...order.map((id) => documentNodes.get(id)!.position.x));
  const authorY = Math.min(...order.map((id) => documentNodes.get(id)!.position.y));
  const authorPositions = Object.fromEntries(
    order.map((id) => {
      const { x, y } = documentNodes.get(id)!.position;

      return [id, { x: (x - authorX) * metrics.authorScale, y: (y - authorY) * metrics.authorScale }];
    })
  );
  const all: WorkflowPreviewLayout =
    same && isSameData(same.all.positions, authorPositions)
      ? same.all
      : {
          edges: same?.all.edges ?? allEdges,
          neighbours: same?.all.neighbours ?? getNeighbours(order, allEdges),
          nodeIds: same?.all.nodeIds ?? order,
          nodeWidth: metrics.nodeWidth,
          pills: {},
          positions: authorPositions,
        };
  const inputs = collectInputs(form);
  const shared = { form, inputGroups: groupInputs(form), inputs, nodes, structureKey };

  const folded = new Map(
    order.flatMap((id) => {
      const input = getFoldableInput(id);

      return input ? [[id, input] as const] : [];
    })
  );
  const hidden = new Set(order.filter(isPlumbing));

  if (folded.size === 0 && hidden.size === 0) {
    return { ...shared, all, simplified: null };
  }

  const visible = order.filter((id) => !folded.has(id) && !hidden.has(id));
  const visibleIds = new Set(visible);

  /** The visible nodes `id` feeds, directly or through hidden plumbing other than a recorder. */
  const reach = (id: string) => {
    const direct = new Set<string>();
    const rerouted = new Set<string>();
    const pending = [id];
    const seen = new Set<string>();

    while (pending.length > 0) {
      const current = pending.pop()!;

      for (const target of outgoing.get(current) ?? []) {
        if (visibleIds.has(target)) {
          (current === id ? direct : rerouted).add(target);
        } else if (
          hidden.has(target) &&
          !RECORDING_TYPES.has(documentNodes.get(target)!.data.type) &&
          !seen.has(target)
        ) {
          seen.add(target);
          pending.push(target);
        }
      }
    }

    return { direct, rerouted };
  };

  const edges: WorkflowPreviewEdge[] = allEdges.filter(
    (edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target)
  );
  const pills: Record<string, WorkflowPreviewPill[]> = {};

  for (const id of order) {
    if (hidden.has(id)) {
      continue;
    }

    const { direct, rerouted } = reach(id);
    const foldedInput = folded.get(id);

    if (foldedInput) {
      // Labelled by the form even while its value is empty: the form names what the value is for.
      const node = documentNodes.get(id)!;
      const template = getTemplate(templates, node.data.type);
      const value = describeValue(node.data.inputs[foldedInput.name]?.value, foldedInput);
      const pill: WorkflowPreviewPill = {
        label: exposedFields.has(fieldKey(id, foldedInput.name))
          ? getFieldLabel(node, template, foldedInput.name)
          : nodes.get(id)!.title,
        nodeId: id,
        value: value && isShortValue(value) ? value : null,
      };

      for (const consumer of [...new Set([...direct, ...rerouted])].sort(byRank)) {
        (pills[consumer] ??= []).push(pill);
      }

      continue;
    }

    for (const target of [...rerouted].sort(byRank)) {
      if (!direct.has(target)) {
        edges.push({ id: `rerouted:${id}:${target}`, isLoopLinkage: false, isRerouted: true, source: id, target });
      }
    }
  }

  if (same?.simplified) {
    const kept = Object.fromEntries(
      Object.entries(pills).map(([consumer, row]) => [consumer, keepSame(same.simplified!.pills[consumer], row)])
    );
    const simplified = isSameData(same.simplified.pills, kept) ? same.simplified : { ...same.simplified, pills: kept };

    return { ...shared, all, simplified };
  }

  const positions = getPreviewLayeredPositions(
    visible.map((id) => ({ id, pillsHeight: (pills[id]?.length ?? 0) * metrics.pillHeight })),
    edges
      .filter((edge) => !edge.isLoopLinkage)
      .map((edge) => ({ sourceNodeId: edge.source, targetNodeId: edge.target })),
    metrics
  );

  return {
    ...shared,
    all,
    simplified: {
      edges,
      neighbours: getNeighbours(visible, edges),
      nodeIds: visible,
      nodeWidth: metrics.nodeWidth,
      pills,
      positions,
    },
  };
};

interface ModelInputs {
  metrics: WorkflowPreviewMetrics;
  templates: InvocationTemplates;
}

/** Per document, the models built for it: one per templates and metrics in use (a drawing's chip width among them). */
const modelCache = new WeakMap<ProjectGraphState, (ModelInputs & { model: WorkflowPreviewModel })[]>();
const MODEL_CACHE_SIZE = 4;
const modelInputs = new WeakMap<WorkflowPreviewModel, ModelInputs>();

/** Cached per document, templates, metrics; a `previous` model's unchanged nodes, pills and layouts stay as-is. */
export const buildWorkflowPreviewModel = (
  document: ProjectGraphState,
  templates: InvocationTemplates,
  metrics: WorkflowPreviewMetrics = WORKFLOW_PREVIEW_METRICS,
  previous: WorkflowPreviewModel | null = null
): WorkflowPreviewModel => {
  const built = modelCache.get(document) ?? [];
  const cached = built.find((entry) => entry.templates === templates && entry.metrics === metrics);

  if (cached) {
    return cached.model;
  }

  const previousInputs = previous ? modelInputs.get(previous) : undefined;
  const model = buildModel(
    document,
    templates,
    metrics,
    previousInputs?.templates === templates && previousInputs.metrics === metrics ? previous : null
  );

  modelCache.set(document, [{ metrics, model, templates }, ...built].slice(0, MODEL_CACHE_SIZE));
  modelInputs.set(model, { metrics, templates });

  return model;
};

export const findWorkflowPreviewNodes = (
  model: WorkflowPreviewModel,
  layout: WorkflowPreviewLayout,
  query: string
): string[] => {
  const needle = query.trim().toLocaleLowerCase();

  if (!needle) {
    return [];
  }

  return layout.nodeIds.filter((id) => {
    const node = model.nodes.get(id)!;

    return node.title.toLocaleLowerCase().includes(needle) || node.type.toLocaleLowerCase().includes(needle);
  });
};

const COMPILED_META_INPUTS = new Set(['id', 'type', 'is_intermediate', 'use_cache']);

/**
 * A compiled graph as a preview document. `overrides` gives the text for values that differ at run time (a fresh seed);
 * fields `getSetLabel` names become the form, labelled as that setting.
 */
export const compiledGraphToPreviewDocument = (
  graph: PreviewGraphLike & { id: string },
  {
    getSetLabel,
    metrics = WORKFLOW_PREVIEW_METRICS,
    overrides,
    positionHints,
  }: {
    getSetLabel?: (nodeId: string, fieldName: string) => string | null;
    metrics?: WorkflowPreviewMetrics;
    overrides?: Readonly<Record<string, Readonly<Record<string, string>>>>;
    positionHints?: Readonly<Record<string, XYPosition>>;
  } = {}
): ProjectGraphState => {
  const depths = getNodeDepths(
    graph.nodes.map((node) => ({ id: node.id })),
    graph.edges
      .filter((edge) => edge.type !== 'loop_linkage')
      .map((edge) => ({ sourceNodeId: edge.sourceNodeId, targetNodeId: edge.targetNodeId }))
  );
  const rows = new Map<number, number>();
  const nextPosition = (id: string): XYPosition => {
    const depth = depths.get(id) ?? 0;
    const row = rows.get(depth) ?? 0;

    rows.set(depth, row + 1);

    return {
      x: (depth * (metrics.nodeWidth + metrics.layerGap)) / metrics.authorScale,
      y: (row * (metrics.nodeHeight + metrics.rowGap)) / metrics.authorScale,
    };
  };
  const elements: Record<string, WorkflowFormElement> = {};
  const children: string[] = [];

  const nodes = graph.nodes.map((node): WorkflowInvocationNode => {
    const inputs: WorkflowInvocationNode['data']['inputs'] = {};

    for (const [name, value] of Object.entries(node.inputs)) {
      if (COMPILED_META_INPUTS.has(name)) {
        continue;
      }

      const setLabel = getSetLabel?.(node.id, name) ?? null;

      inputs[name] = { label: setLabel ?? '', name, value: overrides?.[node.id]?.[name] ?? value };

      if (setLabel !== null) {
        const id = `set:${node.id}:${name}`;

        children.push(id);
        elements[id] = {
          data: { fieldIdentifier: { fieldName: name, nodeId: node.id }, showDescription: false, showShuffle: false },
          id,
          parentId: 'root',
          type: 'node-field',
        };
      }
    }

    return {
      data: {
        inputs,
        isIntermediate: true,
        isOpen: true,
        label: '',
        nodePack: '',
        notes: '',
        type: node.type,
        useCache: true,
        version: '',
      },
      id: node.id,
      position: positionHints?.[node.id] ?? nextPosition(node.id),
      type: 'invocation',
    };
  });

  elements.root = { data: { children, layout: 'column' }, id: 'root', type: 'container' };

  return {
    author: '',
    contact: '',
    description: '',
    edges: graph.edges.map((edge) => ({
      id: edge.id,
      source: edge.sourceNodeId,
      sourceHandle: edge.sourceField,
      target: edge.targetNodeId,
      targetHandle: edge.targetField,
      type: edge.type ?? 'default',
    })),
    form: { elements, rootElementId: 'root' },
    id: graph.id,
    name: graph.label ?? '',
    nodes,
    notes: '',
    tags: '',
    updatedAt: '',
    version: 2,
    workflowVersion: '',
  };
};
