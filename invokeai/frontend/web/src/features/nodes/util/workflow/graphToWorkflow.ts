import * as dagre from '@dagrejs/dagre';
import { logger } from 'app/logging/logger';
import { getStore } from 'app/store/nanostores/store';
import { NODE_WIDTH } from 'features/nodes/types/constants';
import type { FieldInputInstance } from 'features/nodes/types/field';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { buildFieldInputInstance } from 'features/nodes/util/schema/buildFieldInputInstance';
import { t } from 'i18next';
import { forEach } from 'lodash-es';
import type { NonNullableGraph } from 'services/api/types';
import { v4 as uuidv4 } from 'uuid';

/**
 * Converts a graph to a workflow. This is a best-effort conversion and may not be perfect.
 * For example, if a graph references an unknown node type, that node will be skipped.
 * @param graph The graph to convert to a workflow
 * @param autoLayout Whether to auto-layout the nodes using `dagre`. If false, nodes will be simply stacked on top of one another with an offset.
 * @returns The workflow.
 */
export const graphToWorkflow = (graph: NonNullableGraph, autoLayout = true): WorkflowV3 => {
  const invocationTemplates = getStore().getState().nodes.templates;

  if (!invocationTemplates) {
    throw new Error(t('app.storeNotInitialized'));
  }

  // Initialize the workflow
  const workflow: WorkflowV3 = {
    name: '',
    author: '',
    contact: '',
    description: '',
    meta: {
      category: 'user',
      version: '3.0.0',
    },
    notes: '',
    tags: '',
    version: '',
    exposedFields: [],
    edges: [],
    nodes: [],
  };

  // Convert nodes
  forEach(graph.nodes, (node) => {
    const template = invocationTemplates[node.type];

    // Skip missing node templates - this is a best-effort
    if (!template) {
      logger('nodes').warn(`Node type ${node.type} not found in invocationTemplates`);
      return;
    }

    // Build field input instances for each attr
    const inputs: Record<string, FieldInputInstance> = {};

    forEach(node, (value, key) => {
      // Ignore the non-input keys - I think this is all of them?
      if (key === 'id' || key === 'type' || key === 'is_intermediate' || key === 'use_cache') {
        return;
      }

      const inputTemplate = template.inputs[key];

      // Skip missing input templates
      if (!inputTemplate) {
        logger('nodes').warn(`Input ${key} not found in template for node type ${node.type}`);
        return;
      }

      // This _should_ be all we need to do!
      const inputInstance = buildFieldInputInstance(node.id, inputTemplate);
      inputInstance.value = value;
      inputs[key] = inputInstance;
    });

    workflow.nodes.push({
      id: node.id,
      type: 'invocation',
      position: { x: 0, y: 0 }, // we'll do layout later, just need something here
      data: {
        id: node.id,
        type: node.type,
        version: template.version,
        label: '',
        notes: '',
        isOpen: true,
        isIntermediate: node.is_intermediate ?? false,
        useCache: node.use_cache ?? true,
        inputs,
      },
    });
  });

  forEach(graph.edges, (edge) => {
    workflow.edges.push({
      id: uuidv4(), // we don't have edge IDs in the graph
      type: 'default',
      source: edge.source.node_id,
      sourceHandle: edge.source.field,
      target: edge.destination.node_id,
      targetHandle: edge.destination.field,
    });
  });

  if (autoLayout) {
    // Best-effort auto layout via dagre - not perfect but better than nothing
    const dagreGraph = new dagre.graphlib.Graph();
    // `rankdir` and `align` could be tweaked, but it's gonna be janky no matter what we choose
    dagreGraph.setGraph({ rankdir: 'TB', align: 'UL' });
    dagreGraph.setDefaultEdgeLabel(() => ({}));

    // We don't know the dimensions of the nodes until we load the graph into `reactflow` - use a reasonable value
    forEach(graph.nodes, (node) => {
      const width = NODE_WIDTH;
      const height = NODE_WIDTH * 1.5;
      dagreGraph.setNode(node.id, { width, height });
    });

    graph.edges.forEach((edge) => {
      dagreGraph.setEdge(edge.source.node_id, edge.destination.node_id);
    });

    // This does the magic
    dagre.layout(dagreGraph);

    // Update the workflow now that we've got the positions
    workflow.nodes.forEach((node) => {
      const nodeWithPosition = dagreGraph.node(node.id);
      node.position = {
        x: nodeWithPosition.x - nodeWithPosition.width / 2,
        y: nodeWithPosition.y - nodeWithPosition.height / 2,
      };
    });
  } else {
    // Stack nodes with a 50px,50px offset from the previous ndoe
    let x = 0;
    let y = 0;
    workflow.nodes.forEach((node) => {
      node.position = { x, y };
      x = x + 50;
      y = y + 50;
    });
  }

  return workflow;
};
