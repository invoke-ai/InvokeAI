import type { WorkflowBackendGraph } from './graphContracts';
import type {
  FieldInputTemplate,
  FieldOutputTemplate,
  InvocationTemplates,
  InvocationTemplatesSnapshot,
  ProjectGraphState,
  WorkflowInvocationNode,
} from './types';

import { compileProjectGraph, getProjectGraphReadiness } from './buildGraph';
import { getResolvedWorkflowEdgesIndexed } from './connectors';
import { createWorkflowGraphIndex } from './graphIndex';
import { isInvocationNode } from './types';

const READINESS_IMAGE_NAME = 'layer-action-readiness.png';
const SOURCE_NODE_ID = 'layer-workflow-source';
const CAPTURE_NODE_ID = 'layer-workflow-output';

export interface WorkflowImageBinding {
  nodeId: string;
  fieldName: string;
  label: string;
}

export type LayerWorkflowDestination = 'gallery' | 'staging' | 'replace' | 'copy-raster';

export interface LayerWorkflowSelection {
  destination: LayerWorkflowDestination;
  input: WorkflowImageBinding | null;
  output: WorkflowImageBinding | null;
}

export type GetRunnableLayerWorkflowInputs = (output: WorkflowImageBinding) => readonly WorkflowImageBinding[];

export interface BuildLayerWorkflowGraphOptions {
  document: ProjectGraphState;
  templatesSnapshot: InvocationTemplatesSnapshot;
  input: WorkflowImageBinding;
  output: WorkflowImageBinding;
  imageName: string;
}

export interface BuiltLayerWorkflowGraph {
  graph: WorkflowBackendGraph;
  outputNodeId: string;
}

const isSingleImageField = (field: FieldInputTemplate | FieldOutputTemplate): boolean =>
  field.type.name === 'ImageField' && field.type.cardinality === 'SINGLE' && !field.type.batch;

const getNodeLabel = (node: WorkflowInvocationNode, templates: InvocationTemplates): string =>
  node.data.label || templates[node.data.type]?.title || node.data.type;

const getInputLabel = (node: WorkflowInvocationNode, field: FieldInputTemplate): string =>
  node.data.inputs[field.name]?.label || field.title || field.name;

const toBinding = (node: WorkflowInvocationNode, fieldName: string, fieldLabel: string, nodeLabel: string) => ({
  fieldName,
  label: `${nodeLabel} → ${fieldLabel}`,
  nodeId: node.id,
});

const isSameBinding = (left: WorkflowImageBinding, right: WorkflowImageBinding): boolean =>
  left.nodeId === right.nodeId && left.fieldName === right.fieldName;

/** Chooses the first output that can actually run, rather than assuming the first output is runnable. */
export const getDefaultLayerWorkflowSelection = (
  outputs: readonly WorkflowImageBinding[],
  getRunnableInputs: GetRunnableLayerWorkflowInputs
): LayerWorkflowSelection => {
  for (const output of outputs) {
    const input = getRunnableInputs(output)[0];

    if (input) {
      return { destination: 'gallery', input, output };
    }
  }

  return { destination: 'gallery', input: null, output: null };
};

/** Reconciles a changed output while preserving a still-runnable input by field identity. */
export const reconcileLayerWorkflowSelection = (
  current: LayerWorkflowSelection,
  output: WorkflowImageBinding,
  runnableInputs: readonly WorkflowImageBinding[]
): LayerWorkflowSelection => {
  const currentInput = current.input;
  const preservedInput = currentInput
    ? runnableInputs.find((candidate) => isSameBinding(candidate, currentInput))
    : undefined;

  return {
    destination: current.destination,
    input: preservedInput ?? runnableInputs[0] ?? null,
    output,
  };
};

const sortInputFields = (fields: FieldInputTemplate[]): FieldInputTemplate[] =>
  fields
    .map((field, index) => ({ field, index }))
    .sort(
      (left, right) =>
        (left.field.uiOrder ?? Number.MAX_SAFE_INTEGER) - (right.field.uiOrder ?? Number.MAX_SAFE_INTEGER) ||
        left.index - right.index
    )
    .map(({ field }) => field);

/** Returns all unconnected, scalar ImageField inputs in document/UI order. */
export const getLayerWorkflowInputs = (
  document: ProjectGraphState,
  templates: InvocationTemplates
): WorkflowImageBinding[] => {
  const index = createWorkflowGraphIndex(document.nodes, document.edges);
  const connectedInputs = new Set(
    getResolvedWorkflowEdgesIndexed(document.edges, index, templates).map(
      (edge) => `${edge.target}:${edge.targetHandle}`
    )
  );
  const bindings: WorkflowImageBinding[] = [];

  for (const candidate of document.nodes) {
    if (!isInvocationNode(candidate)) {
      continue;
    }

    const template = templates[candidate.data.type];

    if (!template) {
      continue;
    }

    for (const field of sortInputFields(Object.values(template.inputs))) {
      if (!isSingleImageField(field) || connectedInputs.has(`${candidate.id}:${field.name}`)) {
        continue;
      }

      bindings.push(
        toBinding(candidate, field.name, getInputLabel(candidate, field), getNodeLabel(candidate, templates))
      );
    }
  }

  return bindings;
};

/** Returns scalar ImageField outputs in document/field order, excluding the image echo primitive. */
export const getLayerWorkflowOutputs = (
  document: ProjectGraphState,
  templates: InvocationTemplates
): WorkflowImageBinding[] => {
  const bindings: WorkflowImageBinding[] = [];

  for (const candidate of document.nodes) {
    if (!isInvocationNode(candidate) || candidate.data.type === 'image') {
      continue;
    }

    const template = templates[candidate.data.type];

    if (!template) {
      continue;
    }

    for (const field of Object.values(template.outputs)) {
      if (!isSingleImageField(field)) {
        continue;
      }

      bindings.push(toBinding(candidate, field.name, field.title || field.name, getNodeLabel(candidate, templates)));
    }
  }

  return bindings;
};

const getAvailableBinding = (
  bindings: WorkflowImageBinding[],
  selected: WorkflowImageBinding
): WorkflowImageBinding | undefined => bindings.find((candidate) => isSameBinding(candidate, selected));

const allocateNodeId = (graph: WorkflowBackendGraph, base: string): string => {
  if (!graph.nodes[base]) {
    return base;
  }

  let suffix = 1;

  while (graph.nodes[`${base}-${suffix}`]) {
    suffix += 1;
  }

  return `${base}-${suffix}`;
};

/**
 * Builds a throwaway backend graph that feeds one workflow image input and
 * captures one workflow image output. The editable workflow document is never
 * mutated.
 */
export const buildLayerWorkflowGraph = (options: BuildLayerWorkflowGraphOptions): BuiltLayerWorkflowGraph => {
  const { document, imageName, input, output, templatesSnapshot } = options;

  if (templatesSnapshot.status !== 'loaded') {
    throw new Error('Workflow node definitions are not loaded.');
  }

  const templates = templatesSnapshot.templates;
  const cloned = structuredClone(document);
  const availableInput = getAvailableBinding(getLayerWorkflowInputs(cloned, templates), input);

  if (!availableInput) {
    throw new Error('The selected workflow input binding is no longer available.');
  }

  const availableOutput = getAvailableBinding(getLayerWorkflowOutputs(cloned, templates), output);

  if (!availableOutput) {
    throw new Error('The selected workflow output binding is no longer available.');
  }

  const inputNode = cloned.nodes.find(
    (candidate): candidate is WorkflowInvocationNode => candidate.id === input.nodeId && isInvocationNode(candidate)
  );
  const inputTemplate = inputNode ? templates[inputNode.data.type]?.inputs[input.fieldName] : undefined;

  if (!inputNode || !inputTemplate) {
    throw new Error('The selected workflow input binding is no longer available.');
  }

  const isConnectionOnly = inputTemplate.input === 'connection';
  const externallySatisfiedInputs = isConnectionOnly ? new Set([`${input.nodeId}:${input.fieldName}`]) : undefined;

  if (!isConnectionOnly) {
    const instance = inputNode.data.inputs[input.fieldName];

    inputNode.data.inputs[input.fieldName] = {
      label: instance?.label ?? '',
      name: input.fieldName,
      ...(instance?.description === undefined ? {} : { description: instance.description }),
      value: { image_name: imageName },
    };
  }

  const readiness = getProjectGraphReadiness(cloned, templatesSnapshot, { externallySatisfiedInputs });

  if (!readiness.canInvoke) {
    throw new Error(`Workflow is not ready: ${readiness.reasons.join(' ')}`);
  }

  const compiled = compileProjectGraph(cloned, templates);
  const graph = compiled.backendGraph;

  if (!graph) {
    throw new Error('Workflow compilation did not produce a backend graph.');
  }

  for (const graphNode of Object.values(graph.nodes)) {
    graphNode.is_intermediate = true;
  }

  if (isConnectionOnly) {
    const sourceNodeId = allocateNodeId(graph, SOURCE_NODE_ID);
    const targetNode = graph.nodes[input.nodeId];

    if (!targetNode) {
      throw new Error('The selected workflow input node was not compiled.');
    }

    delete targetNode[input.fieldName];
    graph.nodes[sourceNodeId] = {
      id: sourceNodeId,
      image: { image_name: imageName },
      is_intermediate: true,
      type: 'image',
      use_cache: false,
    };
    graph.edges.push({
      destination: { field: input.fieldName, node_id: input.nodeId },
      source: { field: 'image', node_id: sourceNodeId },
    });
  }

  if (!graph.nodes[output.nodeId]) {
    throw new Error('The selected workflow output node was not compiled.');
  }

  const outputNodeId = allocateNodeId(graph, CAPTURE_NODE_ID);

  graph.nodes[outputNodeId] = {
    id: outputNodeId,
    is_intermediate: true,
    type: 'image',
    use_cache: false,
  };
  graph.edges.push({
    destination: { field: 'image', node_id: outputNodeId },
    source: { field: output.fieldName, node_id: output.nodeId },
  });

  return { graph, outputNodeId };
};

/** Keeps only inputs that produce a fully runnable graph after sentinel injection. */
export const getRunnableLayerWorkflowInputs = (
  document: ProjectGraphState,
  templatesSnapshot: InvocationTemplatesSnapshot,
  output: WorkflowImageBinding
): WorkflowImageBinding[] => {
  if (templatesSnapshot.status !== 'loaded') {
    return [];
  }

  return getLayerWorkflowInputs(document, templatesSnapshot.templates).filter((input) => {
    try {
      buildLayerWorkflowGraph({
        document,
        imageName: READINESS_IMAGE_NAME,
        input,
        output,
        templatesSnapshot,
      });
      return true;
    } catch {
      return false;
    }
  });
};
