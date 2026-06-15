import type {
  ContainerFormElement,
  FieldIdentifier,
  InvocationTemplate,
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
  XYPosition,
} from './types';
import { isInvocationNode, isNotesNode } from './types';

/**
 * Pure operations on the project graph document. The workbench reducer
 * delegates all workflow edits to `projectGraphReducer` so the document logic
 * stays testable and `workbenchState.ts` only grows one action case.
 */

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

  return createProjectGraph(legacyId ?? createWorkflowId('project-graph'));
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

// #endregion

// #region Document reducer

export type ProjectGraphAction =
  | { type: 'addNode'; node: WorkflowNode }
  | { type: 'addNodeAndEdge'; node: WorkflowNode; edge: WorkflowEdge }
  | { type: 'addGraphElements'; nodes: WorkflowNode[]; edges: WorkflowEdge[] }
  | { type: 'removeNodes'; nodeIds: string[] }
  | { type: 'setNodePosition'; nodeId: string; position: XYPosition }
  | { type: 'setNodeLabel'; nodeId: string; label: string }
  | { type: 'setNodeNotes'; nodeId: string; notes: string }
  | { type: 'setNodeIsOpen'; nodeId: string; isOpen: boolean }
  | { type: 'setNodeIsIntermediate'; nodeId: string; isIntermediate: boolean }
  | { type: 'setFieldValue'; nodeId: string; fieldName: string; value: unknown }
  | { type: 'setFieldLabel'; nodeId: string; fieldName: string; label: string }
  | { type: 'setFieldDescription'; nodeId: string; fieldName: string; description: string }
  | { type: 'addEdge'; edge: WorkflowEdge }
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
  | { type: 'setContainerLayout'; elementId: string; layout: 'row' | 'column' }
  | { type: 'setMetadata'; patch: Partial<WorkflowMetadata> };

const undoLabels: Partial<Record<ProjectGraphAction['type'], string>> = {
  addEdge: 'Connect workflow fields',
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
  setNodeFieldShowDescription: 'Edit workflow form',
  unexposeField: 'Remove workflow field from form',
};

/** Returns the project-undo label for an action, or null when the edit should not create an undo entry. */
export const getProjectGraphUndoLabel = (action: ProjectGraphAction): string | null => undoLabels[action.type] ?? null;

/** Edits meaningful enough to auto-select the project graph as the invocation source. */
export const isHighConfidenceGraphEdit = (action: ProjectGraphAction): boolean =>
  action.type === 'addNode' ||
  action.type === 'addNodeAndEdge' ||
  action.type === 'addGraphElements' ||
  action.type === 'removeNodes' ||
  action.type === 'addEdge' ||
  action.type === 'removeEdges' ||
  action.type === 'setFieldValue';

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

      return addEdgeToDocument({ ...document, nodes: [...document.nodes, action.node] }, action.edge);
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

      return {
        ...document,
        edges: document.edges.filter((edge) => !removedNodeIds.has(edge.source) && !removedNodeIds.has(edge.target)),
        form: removeNodeFieldElements(document.form, removedNodeIds),
        nodes: document.nodes.filter((node) => !removedNodeIds.has(node.id)),
      };
    }
    case 'setNodePosition': {
      return updateNode(document, action.nodeId, (node) => ({ ...node, position: { ...action.position } }));
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
    case 'setFieldValue': {
      return setFieldInstance(document, action.nodeId, action.fieldName, (instance) => ({
        ...instance,
        value: action.value,
      }));
    }
    case 'setFieldLabel': {
      return setFieldInstance(document, action.nodeId, action.fieldName, (instance) => ({
        ...instance,
        label: action.label,
      }));
    }
    case 'setFieldDescription': {
      // An emptied override falls back to the template description.
      return setFieldInstance(document, action.nodeId, action.fieldName, (instance) => ({
        ...instance,
        description: action.description || undefined,
      }));
    }
    case 'addEdge': {
      return addEdgeToDocument(document, action.edge);
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
          data: { fieldIdentifier: { ...action.fieldIdentifier }, showDescription: false },
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
      const element = document.form.elements[action.elementId];

      if (!element || element.type !== 'node-field') {
        return document;
      }

      return {
        ...document,
        form: {
          ...document.form,
          elements: {
            ...document.form.elements,
            [element.id]: { ...element, data: { ...element.data, showDescription: action.showDescription } },
          },
        },
      };
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
