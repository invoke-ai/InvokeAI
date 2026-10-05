import { SEED_MODES } from '@platform/core/seed';
import { z } from 'zod';

import type {
  FieldInputTemplate,
  ProjectGraphState,
  WorkflowEdge,
  WorkflowFieldInstance,
  WorkflowForm,
  WorkflowFormElement,
  WorkflowNode,
} from './types';

import { createWorkflowForm, createWorkflowId } from './document';

/**
 * Round-trip legacy WorkflowV3 files, metadata, and library records; recover unknown elements and dangling edges
 * as warnings.
 */

const zXYPosition = z.object({ x: z.number().catch(0), y: z.number().catch(0) }).catch({ x: 0, y: 0 });

// Legacy schemas strip seedMode; reimport then defaults to fixed rather than treating the missing extension as
// corruption.
const zFieldInstance = z.object({
  description: z.string().optional().catch(undefined),
  descriptionOverride: z.boolean().optional().catch(undefined),
  label: z.string().catch(''),
  labelOverride: z.boolean().optional().catch(undefined),
  name: z.string(),
  seedMode: z.enum(SEED_MODES).optional().catch(undefined),
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
  data: z
    .looseObject({
      label: z.string().catch('Connector'),
    })
    .catch({ label: 'Connector' }),
  id: z.string().min(1),
  position: zXYPosition,
  type: z.literal('connector'),
});

const zAnyNode = z.union([zInvocationNode, zNotesNode, zCurrentImageNode, zConnectorNode]);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isPersistedFieldInputTemplate = (value: unknown): value is FieldInputTemplate => {
  if (!isRecord(value) || !isRecord(value.type)) {
    return false;
  }

  return (
    typeof value.name === 'string' &&
    typeof value.title === 'string' &&
    typeof value.description === 'string' &&
    typeof value.required === 'boolean' &&
    (value.fieldKind === 'input' || value.fieldKind === 'internal') &&
    (value.input === 'connection' || value.input === 'direct' || value.input === 'any') &&
    typeof value.type.name === 'string' &&
    (value.type.cardinality === 'SINGLE' ||
      value.type.cardinality === 'COLLECTION' ||
      value.type.cardinality === 'SINGLE_OR_COLLECTION') &&
    typeof value.type.batch === 'boolean'
  );
};

const parsePersistedDynamicInputTemplates = (value: unknown): Record<string, FieldInputTemplate> | undefined => {
  if (!isRecord(value)) {
    return undefined;
  }

  const templates: Record<string, FieldInputTemplate> = {};

  for (const [name, template] of Object.entries(value)) {
    if (isPersistedFieldInputTemplate(template)) {
      templates[name] = template;
    }
  }

  return templates;
};

const zWorkflowEdge = z.object({
  id: z.string().catch(''),
  source: z.string().min(1),
  sourceHandle: z.string().min(1),
  target: z.string().min(1),
  targetHandle: z.string().min(1),
  type: z.enum(['default', 'loop_linkage']).catch('default'),
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
    data: z
      .looseObject({
        fieldIdentifier: zFieldIdentifier,
        showDescription: z.boolean().catch(false),
        showShuffle: z.boolean().optional(),
        // The legacy editor keeps its per-element settings (component, bounds, shuffle) here; they
        // are carried through untouched so a workflow saved from webv2 still opens the same there.
        settings: z.looseObject({ showShuffle: z.boolean().optional() }).optional(),
      })
      .transform(({ settings, showShuffle, ...data }) => ({
        ...data,
        ...(settings ? { settings } : {}),
        showShuffle: showShuffle ?? settings?.showShuffle ?? false,
      })),
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
          data: { fieldIdentifier, showDescription: false, showShuffle: false },
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

  for (const rawNode of parsed.data.nodes) {
    const nodeResult = zAnyNode.safeParse(rawNode);

    if (!nodeResult.success) {
      warnings.push('Skipped an unrecognized node.');
      continue;
    }

    const node = nodeResult.data;

    if (node.type === 'connector') {
      nodes.push({
        data: { label: node.data.label },
        id: node.id,
        position: node.position,
        type: 'connector',
      });
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
      const descriptionOverride =
        instance.descriptionOverride === true && (instance.description === undefined || instance.description === '')
          ? false
          : instance.descriptionOverride;
      inputs[name] = {
        description: instance.description,
        ...(descriptionOverride === undefined ? {} : { descriptionOverride }),
        label: instance.label,
        ...(instance.labelOverride === undefined ? {} : { labelOverride: instance.labelOverride }),
        name: instance.name || name,
        ...(instance.seedMode === undefined ? {} : { seedMode: instance.seedMode }),
        value: instance.value,
      };
    }

    const dynamicInputTemplates = parsePersistedDynamicInputTemplates(node.data.dynamicInputTemplates);

    nodes.push({
      data: {
        ...(dynamicInputTemplates ? { dynamicInputTemplates } : {}),
        inputs,
        isIntermediate: node.data.isIntermediate,
        isOpen: node.data.isOpen,
        label: node.data.label,
        nodePack: node.data.nodePack,
        notes: node.data.notes,
        type: node.data.type,
        useCache: node.data.useCache,
        version: node.data.version,
        ...(node.data.type === 'call_saved_workflow'
          ? {
              callSavedWorkflowStatus:
                typeof inputs.workflow_id?.value === 'string' && inputs.workflow_id.value.trim()
                  ? ('loading' as const)
                  : ('ready' as const),
            }
          : {}),
      },
      id: node.id,
      position: node.position,
      type: 'invocation',
    });
  }

  const rawEdges = parsed.data.edges.flatMap((rawEdge) => {
    const edgeResult = zWorkflowEdge.safeParse(rawEdge);

    return edgeResult.success ? [edgeResult.data] : [];
  });

  const nodeIds = new Set(nodes.map((node) => node.id));
  const edges: WorkflowEdge[] = [];

  for (const edge of rawEdges) {
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
      type: edge.type,
    });
  }

  const document: ProjectGraphState = {
    author: parsed.data.author,
    contact: parsed.data.contact,
    description: parsed.data.description,
    edges,
    form: parseForm(parsed.data.form, parsed.data.exposedFields, nodeIds, warnings),
    id: createWorkflowId('workflow'),
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

const serializeInvocationNode = (node: Extract<WorkflowNode, { type: 'invocation' }>) => {
  const { callSavedWorkflowStatus: _callSavedWorkflowStatus, ...data } = structuredClone(node.data);

  return {
    data: { ...data, id: node.id },
    id: node.id,
    position: { ...node.position },
    type: node.type,
  };
};

/**
 * Serializes the document to legacy WorkflowV3 JSON (loadable by the v6 editor and the library backend). The JSON
 * carries a library record id only when the caller names one: a portable file or a new template has none.
 */
export const serializeWorkflowJson = (
  document: ProjectGraphState,
  options: { libraryWorkflowId?: string } = {}
): Record<string, unknown> => ({
  author: document.author,
  contact: document.contact,
  description: document.description,
  edges: document.edges.map((edge) => ({ ...edge })),
  exposedFields: [],
  form: {
    elements: structuredClone(document.form.elements),
    rootElementId: document.form.rootElementId,
  },
  ...(options.libraryWorkflowId ? { id: options.libraryWorkflowId } : {}),
  meta: { category: 'user', version: '3.0.0' },
  name: document.name,
  nodes: document.nodes.map((node) =>
    node.type === 'connector'
      ? {
          data: { ...node.data, id: node.id, isOpen: true, type: 'connector' },
          id: node.id,
          position: { ...node.position },
          type: node.type,
        }
      : node.type === 'notes' || node.type === 'current_image'
        ? {
            data: { ...node.data, id: node.id, isOpen: true },
            id: node.id,
            position: { ...node.position },
            type: node.type,
          }
        : serializeInvocationNode(node)
  ),
  notes: document.notes,
  tags: document.tags,
  version: document.workflowVersion,
});

/** True when the legacy WorkflowWithoutID schema can accept the workflow's output contract. */
export const hasMultipleWorkflowReturnNodes = (document: ProjectGraphState): boolean =>
  document.nodes.filter((node) => node.type === 'invocation' && node.data.type === 'workflow_return').length > 1;

/**
 * Exclude runtime templates and current-image decorations from submitted workflow metadata while preserving
 * recallable loop links.
 */
export const serializeWorkflowJsonForSubmission = (document: ProjectGraphState): Record<string, unknown> => {
  const serialized = serializeWorkflowJson(document);
  const currentImageNodeIds = new Set(
    document.nodes.filter((node) => node.type === 'current_image').map((node) => node.id)
  );
  const nodes = Array.isArray(serialized.nodes) ? serialized.nodes : [];
  const edges = Array.isArray(serialized.edges) ? serialized.edges : [];

  return {
    ...serialized,
    edges: edges
      .filter((edge): edge is Record<string, unknown> => isRecord(edge))
      .filter(
        (edge) =>
          typeof edge.source !== 'string' ||
          typeof edge.target !== 'string' ||
          (!currentImageNodeIds.has(edge.source) && !currentImageNodeIds.has(edge.target))
      ),
    nodes: nodes
      .filter((node): node is Record<string, unknown> => isRecord(node))
      .filter((node) => node.type !== 'current_image')
      .map((node) => {
        if (!isRecord(node.data) || node.type !== 'invocation') {
          return node;
        }

        const { dynamicInputTemplates: _dynamicInputTemplates, ...data } = node.data;
        return node.data.type === 'call_saved_workflow' && isRecord(_dynamicInputTemplates)
          ? { ...node, data: { ...data, dynamicInputTemplates: _dynamicInputTemplates } }
          : { ...node, data };
      }),
  };
};
