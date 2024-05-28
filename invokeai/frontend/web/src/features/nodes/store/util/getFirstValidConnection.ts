import type { Templates } from 'features/nodes/store/types';
import { validateConnection } from 'features/nodes/store/util/validateConnection';
import type { FieldInputTemplate, FieldOutputTemplate } from 'features/nodes/types/field';
import type { AnyNode, InvocationNodeEdge } from 'features/nodes/types/invocation';
import { map } from 'lodash-es';
import type { Connection, Edge } from 'reactflow';

/**
 *
 * @param source The source (node id)
 * @param sourceHandle The source handle (field name), if any
 * @param target The target (node id)
 * @param targetHandle The target handle (field name), if any
 * @param nodes The current nodes
 * @param edges The current edges
 * @param templates The current templates
 * @param edgePendingUpdate The edge pending update, if any
 * @returns
 */
export const getFirstValidConnection = (
  source: string,
  sourceHandle: string | null,
  target: string,
  targetHandle: string | null,
  nodes: AnyNode[],
  edges: InvocationNodeEdge[],
  templates: Templates,
  edgePendingUpdate: Edge | null
): Connection | null => {
  if (source === target) {
    return null;
  }

  if (sourceHandle && targetHandle) {
    return { source, sourceHandle, target, targetHandle };
  }

  if (sourceHandle && !targetHandle) {
    const candidates = getTargetCandidateFields(
      source,
      sourceHandle,
      target,
      nodes,
      edges,
      templates,
      edgePendingUpdate
    );

    const firstCandidate = candidates[0];
    if (!firstCandidate) {
      return null;
    }

    return { source, sourceHandle, target, targetHandle: firstCandidate.name };
  }

  if (!sourceHandle && targetHandle) {
    const candidates = getSourceCandidateFields(
      target,
      targetHandle,
      source,
      nodes,
      edges,
      templates,
      edgePendingUpdate
    );

    const firstCandidate = candidates[0];
    if (!firstCandidate) {
      return null;
    }

    return { source, sourceHandle: firstCandidate.name, target, targetHandle };
  }

  return null;
};

export const getTargetCandidateFields = (
  source: string,
  sourceHandle: string,
  target: string,
  nodes: AnyNode[],
  edges: Edge[],
  templates: Templates,
  edgePendingUpdate: Edge | null
): FieldInputTemplate[] => {
  const sourceNode = nodes.find((n) => n.id === source);
  const targetNode = nodes.find((n) => n.id === target);
  if (!sourceNode || !targetNode) {
    return [];
  }

  const sourceTemplate = templates[sourceNode.data.type];
  const targetTemplate = templates[targetNode.data.type];
  if (!sourceTemplate || !targetTemplate) {
    return [];
  }

  const sourceField = sourceTemplate.outputs[sourceHandle];

  if (!sourceField) {
    return [];
  }

  const targetCandidateFields = map(targetTemplate.inputs).filter((field) => {
    const c = { source, sourceHandle, target, targetHandle: field.name };
    const r = validateConnection(c, nodes, edges, templates, edgePendingUpdate, true);
    return r.isValid;
  });

  return targetCandidateFields;
};

export const getSourceCandidateFields = (
  target: string,
  targetHandle: string,
  source: string,
  nodes: AnyNode[],
  edges: Edge[],
  templates: Templates,
  edgePendingUpdate: Edge | null
): FieldOutputTemplate[] => {
  const targetNode = nodes.find((n) => n.id === target);
  const sourceNode = nodes.find((n) => n.id === source);
  if (!sourceNode || !targetNode) {
    return [];
  }

  const sourceTemplate = templates[sourceNode.data.type];
  const targetTemplate = templates[targetNode.data.type];
  if (!sourceTemplate || !targetTemplate) {
    return [];
  }

  const targetField = targetTemplate.inputs[targetHandle];

  if (!targetField) {
    return [];
  }

  const sourceCandidateFields = map(sourceTemplate.outputs).filter((field) => {
    const c = { source, sourceHandle: field.name, target, targetHandle };
    const r = validateConnection(c, nodes, edges, templates, edgePendingUpdate, true);
    return r.isValid;
  });

  return sourceCandidateFields;
};
