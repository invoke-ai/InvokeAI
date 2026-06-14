import { z } from 'zod';

import { createWorkflowForm, createWorkflowId } from './document';
import type {
  ProjectGraphState,
  WorkflowEdge,
  WorkflowFieldInstance,
  WorkflowForm,
  WorkflowFormElement,
  WorkflowNode,
} from './types';

/**
 * Import/export between the project graph document and the legacy WorkflowV3
 * JSON format — the format used by workflow files, image-embedded workflows,
 * and the backend workflow library. Parsing is tolerant: recoverable problems
 * (unknown elements, dangling edges, connector nodes) become warnings instead
 * of failures.
 */

const zXYPosition = z.object({ x: z.number().catch(0), y: z.number().catch(0) }).catch({ x: 0, y: 0 });

const zFieldInstance = z.object({
  description: z.string().optional().catch(undefined),
  label: z.string().catch(''),
  name: z.string(),
  value: z.unknown().optional(),
});

const zInvocationNode = z.object({
  data: z.looseObject({
    inputs: z.record(z.string(), zFieldInstance).catch({}),
    isIntermediate: z.boolean().catch(true),
    isOpen: z.boolean().catch(true),
    label: z.string().catch(''),
    nodePack: z.string().catch('invokeai'),
    notes: z.string().catch(''),
    type: z.string(),
    useCache: z.boolean().catch(true),
    version: z.string().catch('1.0.0'),
  }),
  id: z.string().min(1),
  position: zXYPosition,
  type: z.literal('invocation'),
});

const zNotesNode = z.object({
  data: z.looseObject({
    label: z.string().catch('Notes'),
    notes: z.string().catch(''),
  }),
  id: z.string().min(1),
  position: zXYPosition,
  type: z.literal('notes'),
});

const zCurrentImageNode = z.object({
  data: z.looseObject({
    label: z.string().catch('Current Image'),
  }),
  id: z.string().min(1),
  position: zXYPosition,
  type: z.literal('current_image'),
});

const zConnectorNode = z.object({
  id: z.string().min(1),
  position: zXYPosition,
  type: z.literal('connector'),
});

const zAnyNode = z.union([zInvocationNode, zNotesNode, zCurrentImageNode, zConnectorNode]);

const zDefaultEdge = z.object({
  id: z.string().catch(''),
  source: z.string().min(1),
  sourceHandle: z.string().min(1),
  target: z.string().min(1),
  targetHandle: z.string().min(1),
  type: z.literal('default').catch('default'),
});

const zFieldIdentifier = z.object({ fieldName: z.string(), nodeId: z.string() });

const zFormElement = z.discriminatedUnion('type', [
  z.object({
    data: z.looseObject({
      children: z.array(z.string()).catch([]),
      layout: z.enum(['column', 'row']).catch('column'),
    }),
    id: z.string(),
    parentId: z.string().optional(),
    type: z.literal('container'),
  }),
  z.object({
    data: z.looseObject({
      fieldIdentifier: zFieldIdentifier,
      showDescription: z.boolean().catch(false),
    }),
    id: z.string(),
    parentId: z.string().optional(),
    type: z.literal('node-field'),
  }),
  z.object({
    data: z.looseObject({ content: z.string().catch('') }),
    id: z.string(),
    parentId: z.string().optional(),
    type: z.literal('heading'),
  }),
  z.object({
    data: z.looseObject({ content: z.string().catch('') }),
    id: z.string(),
    parentId: z.string().optional(),
    type: z.literal('text'),
  }),
  z.object({
    id: z.string(),
    parentId: z.string().optional(),
    type: z.literal('divider'),
  }),
]);

const zWorkflowJson = z.looseObject({
  author: z.string().catch(''),
  contact: z.string().catch(''),
  description: z.string().catch(''),
  edges: z.array(z.unknown()).catch([]),
  exposedFields: z.array(zFieldIdentifier).catch([]),
  // Workflows that predate the form builder store `form: null`; a malformed
  // form also degrades to "absent" rather than failing the whole parse.
  form: z
    .object({
      elements: z.record(z.string(), z.unknown()),
      rootElementId: z.string(),
    })
    .nullish()
    .catch(null),
  id: z.string().optional(),
  name: z.string().catch(''),
  nodes: z.array(z.unknown()).catch([]),
  notes: z.string().catch(''),
  tags: z.string().catch(''),
  version: z.string().catch('1.0.0'),
});

export interface ParsedWorkflow {
  document: ProjectGraphState;
  warnings: string[];
}

/**
 * Resolves edges that pass through legacy connector nodes by walking each
 * connector chain back to its real invocation source.
 */
const resolveConnectorEdges = (
  edges: Array<z.infer<typeof zDefaultEdge>>,
  connectorIds: Set<string>
): { resolved: Array<z.infer<typeof zDefaultEdge>>; dropped: number } => {
  if (connectorIds.size === 0) {
    return { dropped: 0, resolved: edges };
  }

  const findRealSource = (nodeId: string, visited: Set<string>): { source: string; sourceHandle: string } | null => {
    if (visited.has(nodeId)) {
      return null;
    }

    visited.add(nodeId);

    const inboundEdge = edges.find((edge) => edge.target === nodeId);

    if (!inboundEdge) {
      return null;
    }

    if (!connectorIds.has(inboundEdge.source)) {
      return { source: inboundEdge.source, sourceHandle: inboundEdge.sourceHandle };
    }

    return findRealSource(inboundEdge.source, visited);
  };

  let dropped = 0;
  const resolved: Array<z.infer<typeof zDefaultEdge>> = [];

  for (const edge of edges) {
    if (connectorIds.has(edge.target)) {
      continue; // Edges into connectors are consumed by the resolution walk.
    }

    if (!connectorIds.has(edge.source)) {
      resolved.push(edge);
      continue;
    }

    const realSource = findRealSource(edge.source, new Set());

    if (!realSource) {
      dropped += 1;
      continue;
    }

    resolved.push({ ...edge, source: realSource.source, sourceHandle: realSource.sourceHandle });
  }

  return { dropped, resolved };
};

const parseForm = (
  rawForm: z.infer<typeof zWorkflowJson>['form'],
  exposedFields: Array<z.infer<typeof zFieldIdentifier>>,
  nodeIds: Set<string>,
  warnings: string[]
): WorkflowForm => {
  let form: WorkflowForm | null = null;

  if (rawForm) {
    const elements: Record<string, WorkflowFormElement> = {};

    for (const [id, rawElement] of Object.entries(rawForm.elements)) {
      const parsed = zFormElement.safeParse(rawElement);

      if (parsed.success) {
        elements[id] = parsed.data as WorkflowFormElement;
      } else {
        warnings.push(`Skipped an unrecognized form element (${id}).`);
      }
    }

    const root = elements[rawForm.rootElementId];

    if (root?.type === 'container') {
      form = { elements, rootElementId: rawForm.rootElementId };
    } else if (Object.keys(rawForm.elements).length > 0) {
      warnings.push('The workflow form was malformed and has been reset.');
    }
  }

  if (!form) {
    form = createWorkflowForm();

    // Pre-form workflows list exposed fields instead; migrate them into form elements.
    const root = form.elements[form.rootElementId];

    if (root?.type === 'container') {
      for (const fieldIdentifier of exposedFields) {
        const element: WorkflowFormElement = {
          data: { fieldIdentifier, showDescription: false },
          id: createWorkflowId('node-field'),
          parentId: root.id,
          type: 'node-field',
        };

        form.elements[element.id] = element;
        root.data.children.push(element.id);
      }
    }
  }

  // Drop node-field elements that point at nodes which did not survive parsing.
  for (const element of Object.values(form.elements)) {
    if (element.type !== 'node-field' || nodeIds.has(element.data.fieldIdentifier.nodeId)) {
      continue;
    }

    delete form.elements[element.id];

    const parent = element.parentId ? form.elements[element.parentId] : undefined;

    if (parent?.type === 'container') {
      parent.data.children = parent.data.children.filter((childId) => childId !== element.id);
    }

    warnings.push('Removed a form field that referenced a missing node.');
  }

  return form;
};

export const parseWorkflowJson = (raw: unknown): ParsedWorkflow => {
  const parsed = zWorkflowJson.safeParse(raw);

  if (!parsed.success) {
    throw new Error('This file is not a recognizable InvokeAI workflow.');
  }

  const warnings: string[] = [];
  const nodes: WorkflowNode[] = [];
  const connectorIds = new Set<string>();

  for (const rawNode of parsed.data.nodes) {
    const nodeResult = zAnyNode.safeParse(rawNode);

    if (!nodeResult.success) {
      warnings.push('Skipped an unrecognized node.');
      continue;
    }

    const node = nodeResult.data;

    if (node.type === 'connector') {
      connectorIds.add(node.id);
      continue;
    }

    if (node.type === 'notes') {
      nodes.push({
        data: { label: node.data.label, notes: node.data.notes },
        id: node.id,
        position: node.position,
        type: 'notes',
      });
      continue;
    }

    if (node.type === 'current_image') {
      nodes.push({
        data: { label: node.data.label },
        id: node.id,
        position: node.position,
        type: 'current_image',
      });
      continue;
    }

    const inputs: Record<string, WorkflowFieldInstance> = {};

    for (const [name, instance] of Object.entries(node.data.inputs)) {
      inputs[name] = {
        description: instance.description,
        label: instance.label,
        name: instance.name || name,
        value: instance.value,
      };
    }

    nodes.push({
      data: {
        inputs,
        isIntermediate: node.data.isIntermediate,
        isOpen: node.data.isOpen,
        label: node.data.label,
        nodePack: node.data.nodePack,
        notes: node.data.notes,
        type: node.data.type,
        useCache: node.data.useCache,
        version: node.data.version,
      },
      id: node.id,
      position: node.position,
      type: 'invocation',
    });
  }

  if (connectorIds.size > 0) {
    warnings.push(`Flattened ${connectorIds.size} connector node(s); connectors are not part of this editor.`);
  }

  const rawEdges = parsed.data.edges.flatMap((rawEdge) => {
    const edgeResult = zDefaultEdge.safeParse(rawEdge);

    return edgeResult.success ? [edgeResult.data] : [];
  });
  const { dropped, resolved } = resolveConnectorEdges(rawEdges, connectorIds);

  if (dropped > 0) {
    warnings.push(`Dropped ${dropped} connection(s) that could not be resolved through connectors.`);
  }

  const nodeIds = new Set(nodes.map((node) => node.id));
  const edges: WorkflowEdge[] = [];

  for (const edge of resolved) {
    if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) {
      warnings.push('Dropped a connection that referenced a missing node.');
      continue;
    }

    edges.push({
      id: edge.id || createWorkflowId('edge'),
      source: edge.source,
      sourceHandle: edge.sourceHandle,
      target: edge.target,
      targetHandle: edge.targetHandle,
      type: 'default',
    });
  }

  const document: ProjectGraphState = {
    author: parsed.data.author,
    contact: parsed.data.contact,
    description: parsed.data.description,
    edges,
    form: parseForm(parsed.data.form, parsed.data.exposedFields, nodeIds, warnings),
    id: createWorkflowId('project-graph'),
    libraryWorkflowId: parsed.data.id,
    name: parsed.data.name,
    nodes,
    notes: parsed.data.notes,
    tags: parsed.data.tags,
    updatedAt: new Date().toISOString(),
    version: 2,
    workflowVersion: parsed.data.version,
  };

  return { document, warnings };
};

/** Serializes the document to legacy WorkflowV3 JSON (loadable by the v6 editor and the library backend). */
export const serializeWorkflowJson = (document: ProjectGraphState): Record<string, unknown> => ({
  author: document.author,
  contact: document.contact,
  description: document.description,
  edges: document.edges.map((edge) => ({ ...edge })),
  exposedFields: [],
  form: {
    elements: structuredClone(document.form.elements),
    rootElementId: document.form.rootElementId,
  },
  ...(document.libraryWorkflowId ? { id: document.libraryWorkflowId } : {}),
  meta: { category: 'user', version: '3.0.0' },
  name: document.name,
  nodes: document.nodes.map((node) =>
    node.type === 'notes' || node.type === 'current_image'
      ? {
          data: { ...node.data, id: node.id, isOpen: true },
          id: node.id,
          position: { ...node.position },
          type: node.type,
        }
      : {
          data: { ...structuredClone(node.data), id: node.id },
          id: node.id,
          position: { ...node.position },
          type: node.type,
        }
  ),
  notes: document.notes,
  tags: document.tags,
  version: document.workflowVersion,
});
