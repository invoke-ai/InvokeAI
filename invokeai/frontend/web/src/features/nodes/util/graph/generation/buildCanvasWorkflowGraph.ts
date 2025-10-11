import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { CanvasWorkflowState } from 'features/controlLayers/store/canvasWorkflowSlice';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import { $templates } from 'features/nodes/store/nodesSlice';
import type { Templates } from 'features/nodes/store/types';
import type { BoardField } from 'features/nodes/types/common';
import type { BoardFieldInputInstance } from 'features/nodes/types/field';
import { isBoardFieldInputInstance, isBoardFieldInputTemplate } from 'features/nodes/types/field';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import {
  getOriginalAndScaledSizesForOtherModes,
  selectCanvasOutputFields,
  selectPresetModifiedPrompts,
} from 'features/nodes/util/graph/graphBuilderUtils';
import type { AnyInvocation, AnyInvocationInputField, AnyInvocationOutputField } from 'services/api/types';
import { assert } from 'tsafe';

const log = logger('canvas');

type BuildCanvasWorkflowGraphArg = {
  state: RootState;
  manager: CanvasManager;
  workflowState: CanvasWorkflowState;
};

const resolveBoardField = (field: BoardFieldInputInstance, state: RootState): BoardField | undefined => {
  const { value } = field;
  if (value === 'auto' || !value) {
    const autoAddBoardId = selectAutoAddBoardId(state);
    if (autoAddBoardId === 'none') {
      return undefined;
    }
    return { board_id: autoAddBoardId };
  }
  if (value === 'none') {
    return undefined;
  }
  return value;
};

const buildInvocationNodes = (
  workflow: WorkflowV3,
  state: RootState,
  templates: Templates
): Record<string, AnyInvocation> => {
  const invocations: Record<string, AnyInvocation> = {};

  const canvasWorkflowNodes = state.canvasWorkflowNodes.nodes;

  for (const node of workflow.nodes) {
    if (node.type !== 'invocation') {
      continue;
    }
    const { id, data } = node;
    const template = templates[data.type];
    if (!template) {
      log.warn({ nodeId: id, type: data.type }, 'Canvas workflow node template not found; skipping node');
      continue;
    }

    const canvasWorkflowNode = canvasWorkflowNodes.find((n) => n.id === id);
    const nodeInputs = canvasWorkflowNode?.type === 'invocation' ? canvasWorkflowNode.data.inputs : data.inputs;

    const transformedInputs = Object.entries(nodeInputs).reduce<Record<string, unknown>>((acc, [name, field]) => {
      const fieldTemplate = template.inputs[name];
      if (!fieldTemplate) {
        log.warn({ nodeId: id, field: name }, 'Canvas workflow field template not found; skipping field');
        return acc;
      }
      if (isBoardFieldInputTemplate(fieldTemplate) && isBoardFieldInputInstance(field)) {
        acc[name] = resolveBoardField(field, state);
      } else {
        acc[name] = field.value;
      }
      return acc;
    }, {});

    transformedInputs.use_cache = data.useCache;

    invocations[id] = {
      id,
      type: data.type,
      ...transformedInputs,
      is_intermediate: data.isIntermediate,
    } as AnyInvocation;
  }

  return invocations;
};

export const buildCanvasWorkflowGraph = async ({ state, manager, workflowState }: BuildCanvasWorkflowGraphArg) => {
  const { workflow, inputNodeId, outputNodeId } = workflowState;
  assert(workflow && inputNodeId && outputNodeId, 'Canvas workflow is not ready');

  const templates = $templates.get();
  assert(Object.keys(templates).length > 0, 'Invocation templates are not loaded');

  const prompts = selectPresetModifiedPrompts(state);
  const g = new Graph(getPrefixedId('canvas_workflow_graph'));

  const seed = g.addNode({
    id: getPrefixedId('seed'),
    type: 'integer',
  });

  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
    value: prompts.positive,
  });

  const invocations = buildInvocationNodes(workflow, state, templates);

  const { rect } = getOriginalAndScaledSizesForOtherModes(state);
  const rasterAdapters = manager.compositor.getVisibleAdaptersOfType('raster_layer');
  const initialImage = await manager.compositor.getCompositeImageDTO(rasterAdapters, rect, {
    is_intermediate: true,
    silent: true,
  });

  const inputNode = invocations[inputNodeId];
  assert(inputNode, 'Canvas workflow input node missing after parsing');
  (inputNode as Record<string, unknown>).image = { image_name: initialImage.image_name };

  type WorkflowEdge = {
    source: { nodeId: string; handle: AnyInvocationOutputField };
    target: { nodeId: string; handle: AnyInvocationInputField };
  };

  const edges: WorkflowEdge[] = workflow.edges
    .filter((edge) => edge.type === 'default')
    .map((edge) => ({
      source: { nodeId: edge.source, handle: edge.sourceHandle as AnyInvocationOutputField },
      target: { nodeId: edge.target, handle: edge.targetHandle as AnyInvocationInputField },
    }));

  let graphOutputNodeId = outputNodeId;
  const outputInvocation = invocations[outputNodeId];
  if (outputInvocation) {
    const prefixedId = getPrefixedId('canvas_output');
    invocations[prefixedId] = { ...outputInvocation, id: prefixedId } as AnyInvocation;
    delete invocations[outputNodeId];
    edges.forEach((edge) => {
      if (edge.source.nodeId === outputNodeId) {
        edge.source = { ...edge.source, nodeId: prefixedId };
      }
      if (edge.target.nodeId === outputNodeId) {
        edge.target = { ...edge.target, nodeId: prefixedId };
      }
    });
    graphOutputNodeId = prefixedId;
  }

  // Remove explicit values for connected inputs to avoid validation issues.
  edges.forEach(({ target }) => {
    const node = invocations[target.nodeId];
    if (node) {
      delete (node as Record<string, unknown>)[target.handle];
    }
  });

  Object.values(invocations).forEach((invocation) => {
    g.addNode(invocation);
  });

  edges.forEach(({ source, target }) => {
    const sourceNode = invocations[source.nodeId];
    const targetNode = invocations[target.nodeId];
    if (!sourceNode || !targetNode) {
      log.warn({ source, target }, 'Canvas workflow edge references unknown node; skipping edge');
      return;
    }
    g.addEdgeFromObj({
      source: { node_id: source.nodeId, field: source.handle },
      destination: { node_id: target.nodeId, field: target.handle },
    });
  });

  const outputNode = g.getNode(graphOutputNodeId);
  assert(outputNode, 'Canvas workflow output node missing from graph');
  g.updateNode(outputNode, selectCanvasOutputFields(state));

  return {
    g,
    seed,
    positivePrompt,
  };
};
