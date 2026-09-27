import type { SeedMode } from '@platform/core/seed';

import type {
  ContainerFormElement,
  FieldIdentifier,
  InvocationTemplate,
  InvocationTemplates,
  NodeFieldFormElement,
  ProjectGraphState,
  WorkflowCurrentImageNode,
  WorkflowConnectorNode,
  WorkflowEdge,
  WorkflowFieldInstance,
  WorkflowForm,
  WorkflowFormElement,
  WorkflowInvocationNode,
  WorkflowMetadata,
  WorkflowNode,
  WorkflowNotesNode,
  WorkflowSeedFieldAdvance,
  XYPosition,
} from './types';

import { isWorkflowGeneratorVariant } from './batch';
import {
  CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX,
  clearSavedWorkflowDynamicFields,
  setCallSavedWorkflowStatus,
  syncCallSavedWorkflowFields,
  type SavedWorkflowDynamicField,
} from './callSavedWorkflow';
import { getConnectorDeletionSpliceConnections } from './connectors';
import { isInvocationNode, isNotesNode } from './types';

const now = (): string => new Date().toISOString();

export const createWorkflowId = (prefix: string): string =>
  `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

export const createWorkflowForm = (): WorkflowForm => {
  const rootElementId = createWorkflowId('container');
  const root: ContainerFormElement = {
    data: { children: [], layout: 'column' },
    id: rootElementId,
    type: 'container',
  };

  return { elements: { [rootElementId]: root }, rootElementId };
};

export const createProjectGraph = (id: string, name = 'Untitled Workflow'): ProjectGraphState => ({
  author: '',
  contact: '',
  description: '',
  edges: [],
  form: createWorkflowForm(),
  id,
  name,
  nodes: [],
  notes: '',
  tags: '',
  updatedAt: now(),
  version: 2,
  workflowVersion: '1.0.0',
});

export const cloneProjectGraph = (document: ProjectGraphState): ProjectGraphState => structuredClone(document);

/** Accepts any persisted `projectGraph` shape and yields a current document, preserving the id. */
export const normalizeProjectGraph = (candidate: unknown): ProjectGraphState => {
  if (typeof candidate === 'object' && candidate !== null && (candidate as ProjectGraphState).version === 2) {
    return candidate as ProjectGraphState;
  }

  const legacyId =
    typeof (candidate as { id?: unknown } | null)?.id === 'string' ? (candidate as { id: string }).id : null;

  return createProjectGraph(legacyId ?? createWorkflowId('workflow'));
};

export const buildInvocationNode = (template: InvocationTemplate, position: XYPosition): WorkflowInvocationNode => {
  const inputs: Record<string, WorkflowFieldInstance> = {};

  for (const inputTemplate of Object.values(template.inputs)) {
    inputs[inputTemplate.name] = {
      label: '',
      name: inputTemplate.name,
      value: inputTemplate.default === undefined ? undefined : structuredClone(inputTemplate.default),
    };
  }

  return {
    data: {
      inputs,
      isIntermediate: true,
      isOpen: true,
      label: '',
      nodePack: template.nodePack,
      notes: '',
      type: template.type,
      useCache: template.useCache,
      version: template.version,
      ...(template.type === 'call_saved_workflow' ? { callSavedWorkflowStatus: 'ready' as const } : {}),
    },
    id: createWorkflowId(template.type),
    position,
    type: 'invocation',
  };
};

export const buildNotesNode = (position: XYPosition): WorkflowNotesNode => ({
  data: { label: 'Notes', notes: '' },
  id: createWorkflowId('notes'),
  position,
  type: 'notes',
});

export const buildCurrentImageNode = (position: XYPosition): WorkflowCurrentImageNode => ({
  data: { label: 'Current Image' },
  id: createWorkflowId('current-image'),
  position,
  type: 'current_image',
});

export const buildConnectorNode = (position: XYPosition): WorkflowConnectorNode => ({
  data: { label: '' },
  id: createWorkflowId('connector'),
  position,
  type: 'connector',
});

// #region Form manipulation

const getRootContainer = (form: WorkflowForm): ContainerFormElement => {
  const root = form.elements[form.rootElementId];

  if (root?.type === 'container') {
    return root;
  }

  // A malformed form (bad import, old persistence) recovers as an empty root.
  const recovered = createWorkflowForm();

  return recovered.elements[recovered.rootElementId] as ContainerFormElement;
};

const collectDescendantIds = (form: WorkflowForm, elementId: string): string[] => {
  const element = form.elements[elementId];

  if (!element || element.type !== 'container') {
    return [elementId];
  }

  return [elementId, ...element.data.children.flatMap((childId) => collectDescendantIds(form, childId))];
};

const appendToRoot = (form: WorkflowForm, element: WorkflowFormElement): WorkflowForm => {
  const root = getRootContainer(form);
  const rooted = { ...element, parentId: root.id };

  return {
    elements: {
      ...form.elements,
      [root.id]: { ...root, data: { ...root.data, children: [...root.data.children, rooted.id] } },
      [rooted.id]: rooted,
    },
    rootElementId: root.id,
  };
};

const removeFormElement = (form: WorkflowForm, elementId: string): WorkflowForm => {
  if (elementId === form.rootElementId || !form.elements[elementId]) {
    return form;
  }

  const removedIds = new Set(collectDescendantIds(form, elementId));
  const elements: Record<string, WorkflowFormElement> = {};

  for (const [id, element] of Object.entries(form.elements)) {
    if (removedIds.has(id)) {
      continue;
    }

    elements[id] =
      element.type === 'container'
        ? {
            ...element,
            data: { ...element.data, children: element.data.children.filter((childId) => !removedIds.has(childId)) },
          }
        : element;
  }

  return { ...form, elements };
};

/**
 * Reparents an element to `parentId` at `index` (the drag-and-drop primitive).
 * No-ops when the move is impossible: unknown ids, non-container targets, or
 * dropping a container into its own subtree.
 */
const moveFormElementTo = (form: WorkflowForm, elementId: string, parentId: string, index: number): WorkflowForm => {
  const element = form.elements[elementId];
  const nextParent = form.elements[parentId];

  if (
    !element ||
    elementId === form.rootElementId ||
    !nextParent ||
    nextParent.type !== 'container' ||
    collectDescendantIds(form, elementId).includes(parentId)
  ) {
    return form;
  }

  const previousParent = element.parentId ? form.elements[element.parentId] : undefined;

  if (!previousParent || previousParent.type !== 'container') {
    return form;
  }

  const previousIndex = previousParent.data.children.indexOf(elementId);
  const withoutElement = previousParent.data.children.filter((childId) => childId !== elementId);
  // Removing the element above the drop point shifts the target index back one.
  const adjustedIndex =
    previousParent.id === nextParent.id && previousIndex !== -1 && previousIndex < index ? index - 1 : index;
  const targetChildren = previousParent.id === nextParent.id ? withoutElement : [...nextParent.data.children];
  const clampedIndex = Math.min(Math.max(0, adjustedIndex), targetChildren.length);

  targetChildren.splice(clampedIndex, 0, elementId);

  const elements: Record<string, WorkflowFormElement> = {
    ...form.elements,
    [elementId]: { ...element, parentId: nextParent.id },
    [previousParent.id]: { ...previousParent, data: { ...previousParent.data, children: withoutElement } },
  };

  elements[nextParent.id] = {
    ...(elements[nextParent.id] as ContainerFormElement),
    data: { ...(elements[nextParent.id] as ContainerFormElement).data, children: targetChildren },
  };

  return { ...form, elements };
};

const moveFormElement = (form: WorkflowForm, elementId: string, direction: -1 | 1): WorkflowForm => {
  const element = form.elements[elementId];
  const parent = element?.parentId ? form.elements[element.parentId] : getRootContainer(form);

  if (!element || !parent || parent.type !== 'container') {
    return form;
  }

  const index = parent.data.children.indexOf(elementId);
  const nextIndex = index + direction;

  if (index === -1 || nextIndex < 0 || nextIndex >= parent.data.children.length) {
    return form;
  }

  const children = [...parent.data.children];

  children[index] = children[nextIndex] as string;
  children[nextIndex] = elementId;

  return {
    ...form,
    elements: { ...form.elements, [parent.id]: { ...parent, data: { ...parent.data, children } } },
  };
};

export const findNodeFieldElement = (
  form: WorkflowForm,
  fieldIdentifier: FieldIdentifier
): WorkflowFormElement | undefined =>
  Object.values(form.elements).find(
    (element) =>
      element.type === 'node-field' &&
      element.data.fieldIdentifier.nodeId === fieldIdentifier.nodeId &&
      element.data.fieldIdentifier.fieldName === fieldIdentifier.fieldName
  );

export const isFieldExposed = (form: WorkflowForm, fieldIdentifier: FieldIdentifier): boolean =>
  findNodeFieldElement(form, fieldIdentifier) !== undefined;

const removeNodeFieldElements = (form: WorkflowForm, removedNodeIds: Set<string>): WorkflowForm =>
  Object.values(form.elements)
    .filter((element) => element.type === 'node-field' && removedNodeIds.has(element.data.fieldIdentifier.nodeId))
    .reduce((nextForm, element) => removeFormElement(nextForm, element.id), form);

// #region Node updates

export type WorkflowNodeUpdateStatus = 'current' | 'updatable' | 'incompatible' | 'newer';

type Version = [number, number, number];

/** `1.2.3`, with an optional `v`, a missing patch, or a prerelease tag (custom node packs use them). */
const parseVersion = (version: string): Version | null => {
  const match = /^v?(\d+)\.(\d+)(?:\.(\d+))?(?:[-+][\w.-]+)?$/.exec(version.trim());

  return match ? [Number(match[1]), Number(match[2]), Number(match[3] ?? 0)] : null;
};

const compareVersions = (a: Version, b: Version): number => a[0] - b[0] || a[1] - b[1] || a[2] - b[2];

/**
 * Whether a stored node can take its template's version. Only a same-major template merges safely: a template
 * bumps its major when a value-preserving merge would leave the node invalid (the legacy rule).
 */
export const getNodeUpdateStatus = (
  node: WorkflowInvocationNode,
  template: InvocationTemplate
): WorkflowNodeUpdateStatus => {
  if (node.data.type !== template.type) {
    return 'incompatible';
  }

  if (node.data.version === template.version) {
    return 'current';
  }

  const nodeVersion = parseVersion(node.data.version);
  const templateVersion = parseVersion(template.version);

  if (!nodeVersion || !templateVersion || nodeVersion[0] !== templateVersion[0]) {
    return 'incompatible';
  }

  const order = compareVersions(nodeVersion, templateVersion);

  return order === 0 ? 'current' : order > 0 ? 'newer' : 'updatable';
};

/** Node types whose undeclared inputs the backend accepts (`extra='allow'`), so an update must carry them over. */
const EXTRA_INPUT_NODE_TYPES = new Set(['core_metadata']);

interface InvocationNodeUpdate {
  node: WorkflowInvocationNode;
  droppedInputNames: string[];
  /** Inputs whose exposed form elements now belong to another input (`collection` → `images`). */
  renamedInputs: Record<string, string>;
}

/**
 * Fresh template defaults underneath the node's own values: existing labels, values, and node settings win, new
 * inputs start at their defaults, inputs the template no longer declares are dropped (dynamic and extra inputs
 * aside). The id is kept so edges survive.
 */
const updateInvocationNodeToTemplate = (
  node: WorkflowInvocationNode,
  template: InvocationTemplate,
  connectedInputNames: ReadonlySet<string>
): InvocationNodeUpdate => {
  const fresh = buildInvocationNode(template, node.position);
  const inputs: Record<string, WorkflowFieldInstance> = {};
  const renamedInputs: Record<string, string> = {};
  const droppedInputNames: string[] = [];

  for (const [name, freshInstance] of Object.entries(fresh.data.inputs)) {
    const existing = node.data.inputs[name];

    inputs[name] = existing
      ? { ...freshInstance, ...existing, value: existing.value === undefined ? freshInstance.value : existing.value }
      : freshInstance;
  }

  for (const [name, instance] of Object.entries(node.data.inputs)) {
    if (name in inputs) {
      continue;
    }

    if (
      EXTRA_INPUT_NODE_TYPES.has(node.data.type) ||
      node.data.dynamicInputTemplates?.[name] !== undefined ||
      name.startsWith(CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX)
    ) {
      inputs[name] = instance;
    } else {
      droppedInputNames.push(name);
    }
  }

  const sourceVersion = parseVersion(node.data.version);

  // `image_collection` 1.0.2 moved its direct list from `collection` to `images`; a connected `collection` keeps
  // feeding the node and is left alone.
  if (node.data.type === 'image_collection' && sourceVersion && compareVersions(sourceVersion, [1, 0, 2]) < 0) {
    const collection = node.data.inputs.collection;
    const images = inputs.images;

    if (images) {
      renamedInputs.collection = 'images';

      if (
        collection &&
        Array.isArray(collection.value) &&
        !(Array.isArray(images.value) && images.value.length > 0) &&
        !connectedInputNames.has('collection')
      ) {
        inputs.images = { ...images, value: collection.value };

        if (inputs.collection) {
          inputs.collection = { ...inputs.collection, value: fresh.data.inputs.collection?.value };
        }
      }
    }
  }

  // `minimax_h3_denoise` 1.3.0 turned the free frame count into a choice list; a count still on the grid becomes
  // that choice, anything else falls back to the default.
  if (node.data.type === 'minimax_h3_denoise' && sourceVersion && compareVersions(sourceVersion, [1, 3, 0]) < 0) {
    const frames = inputs.num_frames;
    const frameTemplate = template.inputs.num_frames;

    if (frames && frameTemplate?.options && typeof frames.value === 'number') {
      const choice = String(frames.value);

      inputs.num_frames = {
        ...frames,
        value: frameTemplate.options.includes(choice) ? choice : frameTemplate.default,
      };
    }
  }

  return {
    droppedInputNames,
    node: { ...node, data: { ...node.data, inputs, version: template.version } },
    renamedInputs,
  };
};

export interface WorkflowNodesUpdate {
  document: ProjectGraphState;
  updatedNodeIds: string[];
  /** Nodes whose template is newer by a major or older than the node; they need deleting and re-adding. */
  skippedNodeIds: string[];
  /** Edges and form elements that pointed at inputs the updated templates no longer declare. */
  droppedEdgeIds: string[];
  droppedFormElementIds: string[];
}

/** Nodes an `updateNodes` action would move; drives the update-all count and the per-node menu item. */
export const getUpdatableNodeIds = (document: Pick<ProjectGraphState, 'nodes'>, templates: InvocationTemplates) =>
  document.nodes.flatMap((node) => {
    const template = isInvocationNode(node) ? templates[node.data.type] : undefined;

    return isInvocationNode(node) && template && getNodeUpdateStatus(node, template) === 'updatable' ? [node.id] : [];
  });

/**
 * Moves the given nodes (every invocation node by default) to their templates' versions, dropping edges and form
 * elements that pointed at inputs the templates no longer declare. Returns the same document when nothing changed.
 */
export const updateWorkflowNodes = (
  document: ProjectGraphState,
  templates: InvocationTemplates,
  nodeIds?: readonly string[]
): WorkflowNodesUpdate => {
  const wanted = nodeIds ? new Set(nodeIds) : null;
  const updatedNodeIds: string[] = [];
  const skippedNodeIds: string[] = [];
  const droppedByNode = new Map<string, Set<string>>();
  const renamedByNode = new Map<string, Record<string, string>>();
  // One pass over the edges; a per-node scan is quadratic on the large graphs the load path opens.
  const connectedInputsByNode = new Map<string, Set<string>>();

  for (const edge of document.edges) {
    if (edge.targetHandle) {
      const handles = connectedInputsByNode.get(edge.target) ?? new Set<string>();

      handles.add(edge.targetHandle);
      connectedInputsByNode.set(edge.target, handles);
    }
  }

  const nodes = document.nodes.map((node) => {
    if (!isInvocationNode(node) || (wanted && !wanted.has(node.id))) {
      return node;
    }

    const template = templates[node.data.type];

    if (!template) {
      return node;
    }

    const status = getNodeUpdateStatus(node, template);

    if (status === 'current') {
      return node;
    }

    if (status !== 'updatable') {
      skippedNodeIds.push(node.id);
      return node;
    }

    const update = updateInvocationNodeToTemplate(node, template, connectedInputsByNode.get(node.id) ?? new Set());

    updatedNodeIds.push(node.id);

    if (update.droppedInputNames.length > 0) {
      droppedByNode.set(node.id, new Set(update.droppedInputNames));
    }

    if (Object.keys(update.renamedInputs).length > 0) {
      renamedByNode.set(node.id, update.renamedInputs);
    }

    return update.node;
  });

  if (updatedNodeIds.length === 0) {
    return { document, droppedEdgeIds: [], droppedFormElementIds: [], skippedNodeIds, updatedNodeIds };
  }

  const droppedEdgeIds = document.edges
    .filter((edge) => edge.targetHandle && droppedByNode.get(edge.target)?.has(edge.targetHandle))
    .map((edge) => edge.id);
  const droppedEdgeIdSet = new Set(droppedEdgeIds);
  const edges = document.edges.filter((edge) => !droppedEdgeIdSet.has(edge.id));
  const droppedFormElementIds: string[] = [];
  let form = document.form;

  for (const element of Object.values(document.form.elements)) {
    if (element.type !== 'node-field') {
      continue;
    }

    const { fieldName, nodeId } = element.data.fieldIdentifier;
    const renamedTo = renamedByNode.get(nodeId)?.[fieldName];

    if (renamedTo) {
      // The renamed input may already be exposed; one element per field keeps the form lookups unambiguous.
      if (findNodeFieldElement(form, { fieldName: renamedTo, nodeId })) {
        form = removeFormElement(form, element.id);
      } else {
        form = {
          ...form,
          elements: {
            ...form.elements,
            [element.id]: { ...element, data: { ...element.data, fieldIdentifier: { fieldName: renamedTo, nodeId } } },
          },
        };
      }
    } else if (droppedByNode.get(nodeId)?.has(fieldName)) {
      form = removeFormElement(form, element.id);
      droppedFormElementIds.push(element.id);
    }
  }

  return {
    document: { ...document, edges, form, nodes },
    droppedEdgeIds,
    droppedFormElementIds,
    skippedNodeIds,
    updatedNodeIds,
  };
};

// #endregion

export type ProjectGraphAction =
  | { type: 'addNode'; node: WorkflowNode }
  | { type: 'addNodeAndEdge'; node: WorkflowNode; edge: WorkflowEdge | WorkflowEdge[] }
  | { type: 'addGraphElements'; nodes: WorkflowNode[]; edges: WorkflowEdge[] }
  | { type: 'removeNodes'; nodeIds: string[] }
  | { type: 'setNodePosition'; nodeId: string; position: XYPosition }
  | { type: 'setNodeLabel'; nodeId: string; label: string }
  | { type: 'setNodeNotes'; nodeId: string; notes: string }
  | { type: 'setNodeIsOpen'; nodeId: string; isOpen: boolean }
  | { type: 'setNodeIsIntermediate'; nodeId: string; isIntermediate: boolean }
  | { type: 'setNodeUseCache'; nodeId: string; useCache: boolean }
  | { type: 'setFieldValue'; nodeId: string; fieldName: string; value: unknown }
  /** Moves nodes to their templates' versions; the reducer has no template access, so the action carries them. */
  | { type: 'updateNodes'; templates: InvocationTemplates; nodeIds?: readonly string[] }
  | {
      type: 'syncCallSavedWorkflowFields';
      nodeId: string;
      fields: SavedWorkflowDynamicField[];
      edgeIdsToRemove: string[];
      status?: 'loading' | 'ready' | 'error';
    }
  | { type: 'setCallSavedWorkflowStatus'; nodeId: string; status: 'loading' | 'ready' | 'error' }
  | { type: 'retryCallSavedWorkflow'; nodeId: string }
  | { type: 'setFieldLabel'; nodeId: string; fieldName: string; label: string }
  | { type: 'setFieldDescription'; nodeId: string; fieldName: string; description: string }
  | { type: 'setFieldSeedMode'; nodeId: string; fieldName: string; seedMode: SeedMode }
  /** Moves stepping-mode seeds past a queued submission; each field is fenced on the value and mode it was planned from. */
  | { type: 'advanceSeedFields'; advances: readonly WorkflowSeedFieldAdvance[] }
  | { type: 'addEdge'; edge: WorkflowEdge }
  /** Replaces `edgeId` with `edge` as one undoable step (dragging an edge end to a new handle). */
  | { type: 'reconnectEdge'; edgeId: string; edge: WorkflowEdge }
  | { type: 'removeEdges'; edgeIds: string[] }
  | { type: 'exposeField'; fieldIdentifier: FieldIdentifier }
  | { type: 'unexposeField'; fieldIdentifier: FieldIdentifier }
  | { type: 'removeFormElement'; elementId: string }
  | { type: 'moveFormElement'; elementId: string; direction: -1 | 1 }
  | { type: 'moveFormElementTo'; elementId: string; parentId: string; index: number }
  | {
      type: 'addFormElement';
      elementType: 'heading' | 'text' | 'divider' | 'container';
      content?: string;
      layout?: 'row' | 'column';
    }
  | { type: 'setFormElementContent'; elementId: string; content: string }
  | { type: 'setNodeFieldShowDescription'; elementId: string; showDescription: boolean }
  | { type: 'setNodeFieldShowShuffle'; elementId: string; showShuffle: boolean }
  | { type: 'setContainerLayout'; elementId: string; layout: 'row' | 'column' }
  | { type: 'setMetadata'; patch: Partial<WorkflowMetadata> };

const undoLabels: Partial<Record<ProjectGraphAction['type'], string>> = {
  addEdge: 'Connect workflow fields',
  reconnectEdge: 'Reconnect workflow fields',
  addFormElement: 'Edit workflow form',
  addGraphElements: 'Paste workflow nodes',
  addNode: 'Add workflow node',
  addNodeAndEdge: 'Add workflow node',
  exposeField: 'Expose workflow field',
  moveFormElement: 'Edit workflow form',
  moveFormElementTo: 'Edit workflow form',
  removeEdges: 'Disconnect workflow fields',
  removeFormElement: 'Edit workflow form',
  removeNodes: 'Delete workflow nodes',
  setContainerLayout: 'Edit workflow form',
  setFieldDescription: 'Edit workflow field description',
  setFieldLabel: 'Rename workflow field',
  setFieldSeedMode: 'Change workflow seed mode',
  setFieldValue: 'Edit workflow field value',
  setFormElementContent: 'Edit workflow form',
  setMetadata: 'Edit workflow details',
  setNodeFieldShowDescription: 'Edit workflow form',
  setNodeFieldShowShuffle: 'Edit workflow form',
  setNodeIsIntermediate: 'Change workflow node output saving',
  setNodeLabel: 'Rename workflow node',
  setNodeNotes: 'Edit workflow node notes',
  setNodeUseCache: 'Change workflow node caching',
  unexposeField: 'Remove workflow field from form',
  updateNodes: 'Update workflow nodes',
};

export interface ProjectGraphUndoEntry {
  label: string;
  /** Present for edits that arrive as a stream (typing, dragging): consecutive edits with one key fold into one undo step. */
  mergeKey?: string;
}

const isScalarList = (value: unknown): boolean =>
  Array.isArray(value) && value.every((item) => typeof item === 'string' || typeof item === 'number' || item === null);

const getGeneratorVariant = (value: unknown): string | null => {
  const type = typeof value === 'object' && value !== null ? (value as { type?: unknown }).type : undefined;

  return typeof type === 'string' && isWorkflowGeneratorVariant(type) ? type : null;
};

const getUndoMergeKey = (action: ProjectGraphAction): string | undefined => {
  switch (action.type) {
    case 'setFieldDescription':
    case 'setFieldLabel':
      return `${action.type}:${action.nodeId}:${action.fieldName}`;
    case 'setFieldValue':
      // Typed text and dragged numbers stream, as do the rows of a scalar list; a pick (model, board,
      // switch, image list) is one step of its own.
      if (action.fieldName === 'workflow_id') {
        return undefined;
      }

      if (typeof action.value === 'string' || typeof action.value === 'number' || isScalarList(action.value)) {
        return `${action.type}:${action.nodeId}:${action.fieldName}`;
      }

      // Generator settings are typed too; switching the variant starts a new step.
      const variant = getGeneratorVariant(action.value);

      return variant ? `${action.type}:${action.nodeId}:${action.fieldName}:${variant}` : undefined;
    case 'setNodeLabel':
    case 'setNodeNotes':
      return `${action.type}:${action.nodeId}`;
    case 'setFormElementContent':
      return `${action.type}:${action.elementId}`;
    case 'setMetadata':
      return `${action.type}:${Object.keys(action.patch).sort().join(',')}`;
    default:
      return undefined;
  }
};

/** The undo entry an action earns, or null when the edit should not create one (positions, disclosure state, seed advances). */
export const getProjectGraphUndoEntry = (action: ProjectGraphAction): ProjectGraphUndoEntry | null => {
  const label = undoLabels[action.type];

  if (!label) {
    return null;
  }

  const mergeKey = getUndoMergeKey(action);

  return mergeKey ? { label, mergeKey } : { label };
};

const updateNode = (
  document: ProjectGraphState,
  nodeId: string,
  getNode: (node: WorkflowNode) => WorkflowNode
): ProjectGraphState => ({
  ...document,
  nodes: document.nodes.map((node) => (node.id === nodeId ? getNode(node) : node)),
});

const updateInvocationNode = (
  document: ProjectGraphState,
  nodeId: string,
  getNode: (node: WorkflowInvocationNode) => WorkflowInvocationNode
): ProjectGraphState => updateNode(document, nodeId, (node) => (isInvocationNode(node) ? getNode(node) : node));

const setFieldInstance = (
  document: ProjectGraphState,
  nodeId: string,
  fieldName: string,
  getInstance: (instance: WorkflowFieldInstance) => WorkflowFieldInstance
): ProjectGraphState =>
  updateInvocationNode(document, nodeId, (node) => {
    const instance = node.data.inputs[fieldName] ?? { label: '', name: fieldName };

    return {
      ...node,
      data: { ...node.data, inputs: { ...node.data.inputs, [fieldName]: getInstance(instance) } },
    };
  });

const patchNodeFieldElement = (
  document: ProjectGraphState,
  elementId: string,
  patch: Partial<Omit<NodeFieldFormElement['data'], 'fieldIdentifier'>>
): ProjectGraphState => {
  const element = document.form.elements[elementId];

  if (!element || element.type !== 'node-field') {
    return document;
  }

  return {
    ...document,
    form: {
      ...document.form,
      elements: { ...document.form.elements, [element.id]: { ...element, data: { ...element.data, ...patch } } },
    },
  };
};

const addEdgeToDocument = (document: ProjectGraphState, edge: WorkflowEdge): ProjectGraphState => {
  // A non-collect input holds at most one connection; connecting replaces it.
  const targetNode = document.nodes.find((node) => node.id === edge.target);
  const keepsExisting =
    targetNode && isInvocationNode(targetNode) && targetNode.data.type === 'collect' && edge.targetHandle === 'item';
  const edges = keepsExisting
    ? document.edges
    : document.edges.filter(
        (existingEdge) => !(existingEdge.target === edge.target && existingEdge.targetHandle === edge.targetHandle)
      );

  return { ...document, edges: [...edges, edge] };
};

export const projectGraphReducer = (document: ProjectGraphState, action: ProjectGraphAction): ProjectGraphState => {
  const next = applyProjectGraphAction(document, action);

  return next === document ? document : { ...next, updatedAt: now() };
};

const applyProjectGraphAction = (document: ProjectGraphState, action: ProjectGraphAction): ProjectGraphState => {
  switch (action.type) {
    case 'addNode': {
      return { ...document, nodes: [...document.nodes, action.node] };
    }
    case 'addNodeAndEdge': {
      if (document.nodes.some((node) => node.id === action.node.id)) {
        return document;
      }

      const documentWithNode = { ...document, nodes: [...document.nodes, action.node] };
      return (Array.isArray(action.edge) ? action.edge : [action.edge]).reduce(addEdgeToDocument, documentWithNode);
    }
    case 'addGraphElements': {
      if (action.nodes.length === 0 && action.edges.length === 0) {
        return document;
      }

      // Guard against id collisions (a stale paste after an undo): existing
      // ids win, incoming duplicates are dropped along with their edges.
      const existingNodeIds = new Set(document.nodes.map((node) => node.id));
      const nodes = action.nodes.filter((node) => !existingNodeIds.has(node.id));
      const acceptedNodeIds = new Set(nodes.map((node) => node.id));
      const existingEdgeIds = new Set(document.edges.map((edge) => edge.id));
      const edges = action.edges.filter(
        (edge) => !existingEdgeIds.has(edge.id) && acceptedNodeIds.has(edge.source) && acceptedNodeIds.has(edge.target)
      );

      if (nodes.length === 0) {
        return document;
      }

      return { ...document, edges: [...document.edges, ...edges], nodes: [...document.nodes, ...nodes] };
    }
    case 'removeNodes': {
      const removedNodeIds = new Set(action.nodeIds);

      if (removedNodeIds.size === 0) {
        return document;
      }

      const removedConnectorIds = new Set(
        document.nodes.filter((node) => removedNodeIds.has(node.id) && node.type === 'connector').map((node) => node.id)
      );
      const spliceEdges = [...removedConnectorIds].flatMap((connectorId) =>
        getConnectorDeletionSpliceConnections(connectorId, document.nodes, document.edges, removedConnectorIds)
      );
      const remainingNodeIds = new Set(
        document.nodes.filter((node) => !removedNodeIds.has(node.id)).map((node) => node.id)
      );
      const existingEdgeKeys = new Set(
        document.edges.map((edge) => `${edge.source}:${edge.sourceHandle}->${edge.target}:${edge.targetHandle}`)
      );
      const edges = document.edges.filter(
        (edge) => !removedNodeIds.has(edge.source) && !removedNodeIds.has(edge.target)
      );

      for (const edge of spliceEdges) {
        const key = `${edge.source}:${edge.sourceHandle}->${edge.target}:${edge.targetHandle}`;
        if (remainingNodeIds.has(edge.source) && remainingNodeIds.has(edge.target) && !existingEdgeKeys.has(key)) {
          existingEdgeKeys.add(key);
          edges.push(edge);
        }
      }

      return {
        ...document,
        edges,
        form: removeNodeFieldElements(document.form, removedNodeIds),
        nodes: document.nodes.filter((node) => !removedNodeIds.has(node.id)),
      };
    }
    case 'setNodePosition': {
      return updateNode(document, action.nodeId, (node) => ({ ...node, position: { ...action.position } }));
    }
    case 'updateNodes': {
      return updateWorkflowNodes(document, action.templates, action.nodeIds).document;
    }
    case 'setNodeLabel': {
      // Narrowed per branch so TS keeps the node type / data correlation through the spread.
      return updateNode(document, action.nodeId, (node) => {
        if (isInvocationNode(node)) {
          return { ...node, data: { ...node.data, label: action.label } };
        }

        if (isNotesNode(node)) {
          return { ...node, data: { ...node.data, label: action.label } };
        }

        return { ...node, data: { ...node.data, label: action.label } };
      });
    }
    case 'setNodeNotes': {
      return updateNode(document, action.nodeId, (node) => {
        if (isInvocationNode(node)) {
          return { ...node, data: { ...node.data, notes: action.notes } };
        }

        if (isNotesNode(node)) {
          return { ...node, data: { ...node.data, notes: action.notes } };
        }

        return node;
      });
    }
    case 'setNodeIsOpen': {
      return updateInvocationNode(document, action.nodeId, (node) => ({
        ...node,
        data: { ...node.data, isOpen: action.isOpen },
      }));
    }
    case 'setNodeIsIntermediate': {
      return updateInvocationNode(document, action.nodeId, (node) => ({
        ...node,
        data: { ...node.data, isIntermediate: action.isIntermediate },
      }));
    }
    case 'setNodeUseCache': {
      return updateInvocationNode(document, action.nodeId, (node) => ({
        ...node,
        data: { ...node.data, useCache: action.useCache },
      }));
    }
    case 'setFieldValue': {
      const node = document.nodes.find((candidate) => candidate.id === action.nodeId);
      const shouldClearDynamicFields =
        action.fieldName === 'workflow_id' &&
        node &&
        isInvocationNode(node) &&
        node.data.type === 'call_saved_workflow' &&
        node.data.inputs.workflow_id?.value !== action.value;
      const clearedDocument = shouldClearDynamicFields
        ? clearSavedWorkflowDynamicFields(document, action.nodeId)
        : document;
      const nextDocument =
        shouldClearDynamicFields && node && isInvocationNode(node) && node.data.type === 'call_saved_workflow'
          ? setCallSavedWorkflowStatus(
              clearedDocument,
              action.nodeId,
              typeof action.value === 'string' && action.value.trim() ? 'loading' : 'ready'
            )
          : clearedDocument;

      return setFieldInstance(nextDocument, action.nodeId, action.fieldName, (instance) => ({
        ...instance,
        value: action.value,
      }));
    }
    case 'syncCallSavedWorkflowFields': {
      return syncCallSavedWorkflowFields(document, action.nodeId, action.fields, action.edgeIdsToRemove, action.status);
    }
    case 'setCallSavedWorkflowStatus': {
      return setCallSavedWorkflowStatus(document, action.nodeId, action.status);
    }
    case 'retryCallSavedWorkflow': {
      return setCallSavedWorkflowStatus(document, action.nodeId, 'loading');
    }
    case 'setFieldLabel': {
      return setFieldInstance(document, action.nodeId, action.fieldName, (instance) => ({
        ...instance,
        label: action.label,
        labelOverride: action.label !== '',
      }));
    }
    case 'setFieldDescription': {
      return setFieldInstance(document, action.nodeId, action.fieldName, (instance) => ({
        ...instance,
        description: action.description || undefined,
        descriptionOverride: action.description !== '',
      }));
    }
    case 'setFieldSeedMode': {
      // Omit fixed mode to preserve legacy document shape and default behavior.
      return setFieldInstance(document, action.nodeId, action.fieldName, (instance) => {
        const { seedMode: _, ...withoutSeedMode } = instance;
        return action.seedMode === 'fixed' ? withoutSeedMode : { ...instance, seedMode: action.seedMode };
      });
    }
    case 'advanceSeedFields': {
      return action.advances.reduce((next, advance) => {
        const node = next.nodes.find((candidate) => candidate.id === advance.nodeId);
        const instance = node && isInvocationNode(node) ? node.data.inputs[advance.fieldName] : undefined;

        return instance &&
          instance.value === advance.fromSeed &&
          // An absent mode is fixed, which never plans an advance, so a bare compare is the whole fence.
          (instance.seedMode ?? 'fixed') === advance.seedMode
          ? setFieldInstance(next, advance.nodeId, advance.fieldName, (current) => ({
              ...current,
              value: advance.toSeed,
            }))
          : next;
      }, document);
    }
    case 'addEdge': {
      return addEdgeToDocument(document, action.edge);
    }
    case 'reconnectEdge': {
      if (!document.edges.some((edge) => edge.id === action.edgeId)) {
        return document;
      }

      return addEdgeToDocument(
        { ...document, edges: document.edges.filter((edge) => edge.id !== action.edgeId) },
        action.edge
      );
    }
    case 'removeEdges': {
      const removedEdgeIds = new Set(action.edgeIds);

      if (removedEdgeIds.size === 0) {
        return document;
      }

      return { ...document, edges: document.edges.filter((edge) => !removedEdgeIds.has(edge.id)) };
    }
    case 'exposeField': {
      if (isFieldExposed(document.form, action.fieldIdentifier)) {
        return document;
      }

      return {
        ...document,
        form: appendToRoot(document.form, {
          data: { fieldIdentifier: { ...action.fieldIdentifier }, showDescription: false, showShuffle: false },
          id: createWorkflowId('node-field'),
          type: 'node-field',
        }),
      };
    }
    case 'unexposeField': {
      const element = findNodeFieldElement(document.form, action.fieldIdentifier);

      if (!element) {
        return document;
      }

      return { ...document, form: removeFormElement(document.form, element.id) };
    }
    case 'removeFormElement': {
      return { ...document, form: removeFormElement(document.form, action.elementId) };
    }
    case 'moveFormElement': {
      const form = moveFormElement(document.form, action.elementId, action.direction);

      return form === document.form ? document : { ...document, form };
    }
    case 'moveFormElementTo': {
      const form = moveFormElementTo(document.form, action.elementId, action.parentId, action.index);

      return form === document.form ? document : { ...document, form };
    }
    case 'addFormElement': {
      const element: WorkflowFormElement =
        action.elementType === 'divider'
          ? { id: createWorkflowId('divider'), type: 'divider' }
          : action.elementType === 'container'
            ? {
                data: { children: [], layout: action.layout ?? 'column' },
                id: createWorkflowId('container'),
                type: 'container',
              }
            : {
                data: { content: action.content ?? '' },
                id: createWorkflowId(action.elementType),
                type: action.elementType,
              };

      return { ...document, form: appendToRoot(document.form, element) };
    }
    case 'setFormElementContent': {
      const element = document.form.elements[action.elementId];

      if (!element || (element.type !== 'heading' && element.type !== 'text')) {
        return document;
      }

      return {
        ...document,
        form: {
          ...document.form,
          elements: {
            ...document.form.elements,
            [element.id]: { ...element, data: { ...element.data, content: action.content } },
          },
        },
      };
    }
    case 'setContainerLayout': {
      const element = document.form.elements[action.elementId];

      if (!element || element.type !== 'container') {
        return document;
      }

      return {
        ...document,
        form: {
          ...document.form,
          elements: {
            ...document.form.elements,
            [element.id]: { ...element, data: { ...element.data, layout: action.layout } },
          },
        },
      };
    }
    case 'setNodeFieldShowDescription': {
      return patchNodeFieldElement(document, action.elementId, { showDescription: action.showDescription });
    }
    case 'setNodeFieldShowShuffle': {
      const settings = document.form.elements[action.elementId];
      const legacySettings = settings?.type === 'node-field' ? settings.data.settings : undefined;

      return patchNodeFieldElement(document, action.elementId, {
        showShuffle: action.showShuffle,
        ...(legacySettings ? { settings: { ...legacySettings, showShuffle: action.showShuffle } } : {}),
      });
    }
    case 'setMetadata': {
      return { ...document, ...action.patch };
    }
  }
};

// #endregion

/** Ordered, render-ready view of the form tree starting at the root. */
export const getFormChildren = (form: WorkflowForm, containerId?: string): WorkflowFormElement[] => {
  const container = form.elements[containerId ?? form.rootElementId];

  if (!container || container.type !== 'container') {
    return [];
  }

  return container.data.children.flatMap((childId) => {
    const child = form.elements[childId];

    return child ? [child] : [];
  });
};
