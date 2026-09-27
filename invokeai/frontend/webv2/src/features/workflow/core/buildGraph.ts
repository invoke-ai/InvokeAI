import { isSeedMode, planSeedSubmission, SEED_MAX, type SeedMode, wrapSeed } from '@platform/core/seed';

import type { CompiledWorkflowGraph, WorkflowBackendGraph } from './graphContracts';
import type {
  FieldInputTemplate,
  InvocationTemplates,
  InvocationTemplatesSnapshot,
  ProjectGraphState,
  WorkflowFieldInstance,
  WorkflowInvocationNode,
  WorkflowSeedFieldAdvance,
} from './types';

import {
  getWorkflowBatchCollectionField,
  isWorkflowBatchNodeType,
  isWorkflowGeneratorNodeType,
  planWorkflowBatch,
  WORKFLOW_BATCH_MAX_ITEMS,
  type WorkflowBatchDatum,
  type WorkflowGeneratorResolutions,
} from './batch';
import { CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX } from './callSavedWorkflow';
import { createWorkflowId } from './document';
import { getWorkflowFieldInvalidReason, isDirectInputField, isWorkflowCollectionItemValid } from './fields';
import {
  createForLoopValidationReason,
  ForLoopGraphValidationError,
  getCanonicalWorkflowEdges,
  validateForLoopGraph,
  type ForLoopValidationReason,
} from './forLoops';
import { isInvocationNode } from './types';
import { hasAnyCycle } from './validation';

/**
 * Compile documents to immutable queue GraphContract with connector resolution. Batch and generator nodes never
 * reach the backend graph: the batch planner turns them into queue batch groups instead.
 */

export const isExecutableInvocationType = (type: string): boolean =>
  !isWorkflowBatchNodeType(type) && !isWorkflowGeneratorNodeType(type);

const getExecutableNodes = (document: ProjectGraphState): WorkflowInvocationNode[] =>
  document.nodes.filter(
    (node): node is WorkflowInvocationNode => isInvocationNode(node) && isExecutableInvocationType(node.data.type)
  );

const isMissingValue = (value: unknown): boolean => value === undefined || value === null;

const isEmptyValue = (value: unknown): boolean =>
  isMissingValue(value) || (typeof value === 'string' && value.trim() === '');

const getNodeDisplayName = (node: WorkflowInvocationNode, templates: InvocationTemplates): string =>
  node.data.label || templates[node.data.type]?.title || node.data.type;

const getNodeInputTemplates = (
  node: WorkflowInvocationNode,
  template: InvocationTemplates[string]
): FieldInputTemplate[] => Object.values({ ...template.inputs, ...node.data.dynamicInputTemplates });

const toBoardGraphValue = (value: unknown): unknown => {
  if (isEmptyValue(value) || value === 'auto' || value === 'none') {
    return undefined;
  }

  return value;
};

export interface ProjectGraphReadiness {
  canInvoke: boolean;
  reasons: Array<string | ForLoopValidationReason>;
  /** Sessions one run produces through batch nodes; size is null while an async generator is unresolved. */
  batch: { size: number | null } | null;
}

export interface ProjectGraphReadinessOptions {
  /** Required connection inputs supplied by an ephemeral caller after document compilation. */
  externallySatisfiedInputs?: ReadonlySet<string>;
  /** Runs the submission would make; with batch nodes the product must fit the queue. */
  batchCount?: number;
}

/** The reason a batch of `sessions` sessions cannot be queued, or null while it fits. */
export const getWorkflowBatchCapReason = (sessions: number): string | null =>
  sessions > WORKFLOW_BATCH_MAX_ITEMS
    ? `This batch would queue ${sessions.toLocaleString('en-US')} sessions; the queue accepts at most ${WORKFLOW_BATCH_MAX_ITEMS.toLocaleString('en-US')}.`
    : null;

export const getProjectGraphReadiness = (
  document: ProjectGraphState,
  templatesSnapshot: InvocationTemplatesSnapshot,
  options: ProjectGraphReadinessOptions = {}
): ProjectGraphReadiness => {
  if (templatesSnapshot.status === 'error') {
    return { batch: null, canInvoke: false, reasons: ['Node definitions failed to load from the backend.'] };
  }

  if (templatesSnapshot.status !== 'loaded') {
    return { batch: null, canInvoke: false, reasons: ['Node definitions are still loading.'] };
  }

  const templates = templatesSnapshot.templates;
  const invocationNodes = document.nodes.filter(isInvocationNode);
  const canonicalEdges = getCanonicalWorkflowEdges(document);

  if (getExecutableNodes(document).length === 0) {
    return {
      batch: null,
      canInvoke: false,
      reasons: ['The project graph has no nodes. Add nodes in the Workflow view.'],
    };
  }

  const reasons: Array<string | ForLoopValidationReason> = [];
  const connectedInputs = new Set(
    canonicalEdges.map((edge) => `${edge.destination.node_id}:${edge.destination.field}`)
  );

  for (const node of invocationNodes) {
    const template = templates[node.data.type];

    if (!template) {
      reasons.push(`Unknown node type "${node.data.type}".`);
      continue;
    }

    // The batch planner judges a batch node's own list (size, connections); the field rules would only repeat it.
    const batchCollectionField = getWorkflowBatchCollectionField(node.data.type);

    if (node.data.type === 'call_saved_workflow') {
      const workflowId = node.data.inputs.workflow_id?.value;

      if (typeof workflowId !== 'string' || workflowId.trim() === '') {
        reasons.push('Call Saved Workflow requires a saved workflow.');
        continue;
      }

      if (node.data.callSavedWorkflowStatus === 'loading' || node.data.callSavedWorkflowStatus === undefined) {
        reasons.push('Call Saved Workflow inputs are still loading.');
        continue;
      }

      if (node.data.callSavedWorkflowStatus === 'error') {
        reasons.push('The selected saved workflow is unavailable or incompatible.');
        continue;
      }
    }

    for (const inputTemplate of getNodeInputTemplates(node, template)) {
      if (connectedInputs.has(`${node.id}:${inputTemplate.name}`)) {
        continue;
      }

      // A batch node's own list: the planner judges its size, the item rules still judge each entry.
      if (inputTemplate.name === batchCollectionField) {
        const list = node.data.inputs[inputTemplate.name]?.value;

        if (Array.isArray(list) && !list.every((item) => isWorkflowCollectionItemValid(inputTemplate, item))) {
          reasons.push(`"${getNodeDisplayName(node, templates)}" has invalid input "${inputTemplate.title}".`);
        }

        continue;
      }

      if (inputTemplate.input === 'connection') {
        if (inputTemplate.required && !options.externallySatisfiedInputs?.has(`${node.id}:${inputTemplate.name}`)) {
          reasons.push(
            `"${getNodeDisplayName(node, templates)}" is missing a connection for "${inputTemplate.title}".`
          );
        }
        continue;
      }

      const invalidReason = getWorkflowFieldInvalidReason({
        isConnected: false,
        template: inputTemplate,
        value: node.data.inputs[inputTemplate.name]?.value,
      });

      if (inputTemplate.required && isMissingValue(node.data.inputs[inputTemplate.name]?.value)) {
        reasons.push(`"${getNodeDisplayName(node, templates)}" is missing required input "${inputTemplate.title}".`);
      } else if (invalidReason) {
        reasons.push(`"${getNodeDisplayName(node, templates)}" has invalid input "${inputTemplate.title}".`);
      }
    }
  }

  if (
    hasAnyCycle(
      document.nodes,
      canonicalEdges.map((edge) => ({
        id: edge.id,
        source: edge.source.node_id,
        sourceHandle: edge.source.field,
        target: edge.destination.node_id,
        targetHandle: edge.destination.field,
        type: edge.type,
      }))
    )
  ) {
    reasons.push('The project graph contains a cycle.');
  }

  const forLoopError = validateForLoopGraph(document);

  if (forLoopError) {
    reasons.push(createForLoopValidationReason(forLoopError));
  }

  const batchPlan = planWorkflowBatch(document, templates);

  reasons.push(...batchPlan.reasons);

  if (batchPlan.hasBatchNodes && batchPlan.batchSize !== null && options.batchCount !== undefined) {
    const capReason = getWorkflowBatchCapReason(batchPlan.batchSize * options.batchCount);

    if (capReason) {
      reasons.push(capReason);
    }
  }

  return {
    batch: batchPlan.hasBatchNodes ? { size: batchPlan.batchSize } : null,
    canInvoke: reasons.length === 0,
    reasons,
  };
};

const toGraphInputValue = (
  inputTemplate: FieldInputTemplate,
  value: unknown,
  options: { preserveBoardSentinel?: boolean } = {}
): unknown => {
  if (inputTemplate.type.name === 'BoardField') {
    return options.preserveBoardSentinel ? value : toBoardGraphValue(value);
  }

  return value;
};

/** Compiles the document into a `GraphContract` carrying the executable backend graph. */
export const compileProjectGraph = (
  document: ProjectGraphState,
  templates: InvocationTemplates
): CompiledWorkflowGraph => {
  const forLoopError = validateForLoopGraph(document);

  if (forLoopError) {
    throw new ForLoopGraphValidationError(forLoopError);
  }

  const executableNodes = getExecutableNodes(document).filter((node) => templates[node.data.type] !== undefined);
  const executableNodeIds = new Set(executableNodes.map((node) => node.id));
  const backendGraph: WorkflowBackendGraph = { edges: [], id: createWorkflowId('workflow-graph'), nodes: {} };
  const resolvedEdges = getCanonicalWorkflowEdges(document);

  for (const node of executableNodes) {
    const template = templates[node.data.type] as NonNullable<(typeof templates)[string]>;
    const graphNode: Record<string, unknown> = {
      id: node.id,
      is_intermediate: node.data.isIntermediate,
      type: node.data.type,
      use_cache: node.data.useCache,
    };

    const workflowInputs: Record<string, unknown> = {};

    for (const instance of Object.values(node.data.inputs)) {
      const inputTemplate = node.data.dynamicInputTemplates?.[instance.name] ?? template.inputs[instance.name];

      if (!inputTemplate || instance.value === undefined) {
        continue;
      }

      const isSavedWorkflowInput =
        node.data.type === 'call_saved_workflow' && instance.name.startsWith(CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX);
      const value = toGraphInputValue(inputTemplate, instance.value, {
        preserveBoardSentinel: isSavedWorkflowInput,
      });

      if (value !== undefined) {
        if (isSavedWorkflowInput) {
          workflowInputs[instance.name] = value;
        } else {
          graphNode[instance.name] = value;
        }
      }
    }

    if (node.data.type === 'call_saved_workflow') {
      graphNode.workflow_inputs = workflowInputs;
    }

    backendGraph.nodes[node.id] = graphNode as WorkflowBackendGraph['nodes'][string];
  }

  const seenEdgeKeys = new Set<string>();

  for (const edge of resolvedEdges) {
    if (!executableNodeIds.has(edge.source.node_id) || !executableNodeIds.has(edge.destination.node_id)) {
      continue;
    }

    const key = `${edge.type}:${edge.source.node_id}:${edge.source.field}->${edge.destination.node_id}:${edge.destination.field}`;

    if (seenEdgeKeys.has(key)) {
      continue;
    }

    seenEdgeKeys.add(key);
    backendGraph.edges.push({
      destination: edge.destination,
      source: edge.source,
      type: edge.type,
    });

    // A connected input always wins over a stale direct value; sending both
    // would let pydantic reject the node on the ignored direct value.
    const targetNode = backendGraph.nodes[edge.destination.node_id];

    if (targetNode) {
      if (
        targetNode.type === 'call_saved_workflow' &&
        edge.destination.field.startsWith(CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX)
      ) {
        const workflowInputs = targetNode.workflow_inputs;

        if (workflowInputs && typeof workflowInputs === 'object') {
          delete (workflowInputs as Record<string, unknown>)[edge.destination.field];
        }
      } else {
        delete targetNode[edge.destination.field];
      }
    }
  }

  return {
    backendGraph,
    edges: resolvedEdges
      .filter((edge) => executableNodeIds.has(edge.source.node_id) && executableNodeIds.has(edge.destination.node_id))
      .map((edge) => ({
        id: edge.id,
        sourceField: edge.source.field,
        sourceNodeId: edge.source.node_id,
        targetField: edge.destination.field,
        targetNodeId: edge.destination.node_id,
        type: edge.type,
      })),
    id: backendGraph.id,
    label: document.name || 'Workflow',
    nodes: executableNodes.map((node) => ({
      id: node.id,
      inputs: Object.fromEntries(Object.values(node.data.inputs).map((instance) => [instance.name, instance.value])),
      type: node.data.type,
    })),
    updatedAt: new Date().toISOString(),
    version: 1,
  };
};

/**
 * Seed modes require a template-declared scalar seed with the full range. Keep seed arithmetic here so shared
 * field utilities remain lightweight.
 */
export const isSeedInputField = (template: FieldInputTemplate): boolean =>
  template.name === 'seed' &&
  template.type.name === 'IntegerField' &&
  template.type.cardinality === 'SINGLE' &&
  // The modes walk and wrap over 0…SEED_MAX in steps of one, so the template has to
  // accept every value on that walk; a tighter range or step keeps its plain control.
  template.maximum === SEED_MAX &&
  ((template.minimum ?? null) === null || (template.minimum ?? 0) <= 0) &&
  (template.exclusiveMinimum ?? null) === null &&
  (template.exclusiveMaximum ?? null) === null &&
  ((template.multipleOf ?? null) === null || template.multipleOf === 1) &&
  isDirectInputField(template);

export const getWorkflowFieldSeedMode = (instance: Pick<WorkflowFieldInstance, 'seedMode'> | undefined): SeedMode =>
  isSeedMode(instance?.seedMode) ? instance.seedMode : 'fixed';

/** One seed input that varies between runs: the seed the first run uses and the direction of the rest. */
export interface WorkflowSeedAssignment {
  fieldName: string;
  nodeId: string;
  seed: number;
  seedStep: -1 | 1;
}

export interface WorkflowSeedPlan {
  /** Every unconnected seed input in a varying mode, with its first seed; fixed inputs are absent. */
  seeds: WorkflowSeedAssignment[];
  /** Stepping-mode fields to move once the submission is reserved. */
  seedAdvances: WorkflowSeedFieldAdvance[];
}

/**
 * Choose seed starts once per submission and expand runs deterministically at send time. Random preserves the
 * entered seed; stepping reports the next authored value.
 */
export const planWorkflowSeeds = (
  document: ProjectGraphState,
  templates: InvocationTemplates,
  batchCount: number,
  batchSize = 1
): WorkflowSeedPlan => {
  const connectedInputs = new Set(
    getCanonicalWorkflowEdges(document).map((edge) => `${edge.destination.node_id}:${edge.destination.field}`)
  );
  const seeds: WorkflowSeedAssignment[] = [];
  const seedAdvances: WorkflowSeedFieldAdvance[] = [];

  for (const node of getExecutableNodes(document)) {
    const template = templates[node.data.type];

    if (!template) {
      continue;
    }

    for (const inputTemplate of Object.values({ ...template.inputs, ...node.data.dynamicInputTemplates })) {
      if (!isSeedInputField(inputTemplate) || connectedInputs.has(`${node.id}:${inputTemplate.name}`)) {
        continue;
      }

      const instance = node.data.inputs[inputTemplate.name];
      const seedMode = getWorkflowFieldSeedMode(instance);

      if (seedMode === 'fixed') {
        continue;
      }

      const authoredSeed =
        typeof instance?.value === 'number'
          ? instance.value
          : typeof inputTemplate.default === 'number'
            ? inputTemplate.default
            : 0;
      const startSeed = seedMode === 'random' ? Math.floor(Math.random() * SEED_MAX) : wrapSeed(authoredSeed);
      // Every session gets its own seed, as ComfyUI's per-execution advance does, so the walk spans the batch.
      const plan = planSeedSubmission({
        batchCount,
        promptCount: batchSize,
        seedBehaviour: 'per-image',
        seedMode,
        startSeed,
      });

      seeds.push({
        fieldName: inputTemplate.name,
        nodeId: node.id,
        seed: startSeed,
        seedStep: seedMode === 'decrement' ? -1 : 1,
      });

      if (plan.nextSeed !== null) {
        seedAdvances.push({
          fieldName: inputTemplate.name,
          ...(typeof instance?.value === 'number' ? { fromSeed: instance.value } : {}),
          nodeId: node.id,
          seedMode,
          toSeed: plan.nextSeed,
        });
      }
    }
  }

  return { seedAdvances, seeds };
};

/** Writes each planned first seed into the compiled graph, so a single run needs no batch data. */
export const applyWorkflowSeeds = (
  graph: CompiledWorkflowGraph,
  seeds: readonly WorkflowSeedAssignment[]
): CompiledWorkflowGraph => {
  for (const { fieldName, nodeId, seed } of seeds) {
    const backendNode = graph.backendGraph.nodes[nodeId];
    const node = graph.nodes.find((candidate) => candidate.id === nodeId);

    if (backendNode) {
      backendNode[fieldName] = seed;
    }

    if (node) {
      node.inputs[fieldName] = seed;
    }
  }

  return graph;
};

export interface WorkflowSubmissionPlan extends WorkflowSeedPlan {
  /** Runs the submission produces. */
  batchCount: number;
  /** Batch-node groups, in the backend's product-of-zips shape; empty without batch nodes. */
  batchData: WorkflowBatchDatum[][];
  /** Sessions one run produces through batch nodes. */
  batchSize: number;
  graph: CompiledWorkflowGraph;
}

export interface WorkflowSubmissionPlanOptions {
  /** Runs per submission; already sanitized to a positive integer by the caller. */
  batchCount: number;
  /** Async generator outputs resolved by the submitter; the plan is null while any are still pending. */
  generators?: WorkflowGeneratorResolutions;
  random?: () => number;
}

/**
 * Compiles the document with every planned first seed in place and reports the seeds that vary. Returns null when
 * the batch plan is not ready: an unresolved generator, a batch reason, or more sessions than the queue accepts.
 */
export const planWorkflowSubmission = (
  document: ProjectGraphState,
  templates: InvocationTemplates,
  { batchCount, generators, random }: WorkflowSubmissionPlanOptions
): WorkflowSubmissionPlan | null => {
  const batchPlan = planWorkflowBatch(document, templates, { generators, random });

  if (
    batchPlan.reasons.length > 0 ||
    batchPlan.pendingGenerators.length > 0 ||
    batchPlan.batchSize === null ||
    batchPlan.batchSize * batchCount > WORKFLOW_BATCH_MAX_ITEMS
  ) {
    return null;
  }

  const seedPlan = planWorkflowSeeds(document, templates, batchCount, batchPlan.batchSize);

  return {
    ...seedPlan,
    batchCount,
    batchData: batchPlan.groups,
    batchSize: batchPlan.batchSize,
    graph: applyWorkflowSeeds(compileProjectGraph(document, templates), seedPlan.seeds),
  };
};
