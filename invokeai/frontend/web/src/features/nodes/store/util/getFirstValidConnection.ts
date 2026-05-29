import type { Connection } from '@xyflow/react';
import { map } from 'es-toolkit/compat';
import type { Templates } from 'features/nodes/store/types';
import {
  CONNECTOR_INPUT_HANDLE,
  CONNECTOR_OUTPUT_HANDLE,
  resolveConnectorSourceFieldType,
} from 'features/nodes/store/util/connectorTopology';
import { validateConnection } from 'features/nodes/store/util/validateConnection';
import type { FieldInputTemplate, FieldOutputTemplate } from 'features/nodes/types/field';
import type { AnyEdge, AnyNode } from 'features/nodes/types/invocation';
import { isConnectorNode } from 'features/nodes/types/invocation';

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
  edges: AnyEdge[],
  templates: Templates,
  edgePendingUpdate: AnyEdge | null
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
  edges: AnyEdge[],
  templates: Templates,
  edgePendingUpdate: AnyEdge | null
): FieldInputTemplate[] => {
  const sourceNode = nodes.find((n) => n.id === source);
  const targetNode = nodes.find((n) => n.id === target);
  if (!sourceNode || !targetNode) {
    return [];
  }

  if (isConnectorNode(targetNode)) {
    const candidate = {
      name: CONNECTOR_INPUT_HANDLE,
      title: 'Connector Input',
      description: '',
      fieldKind: 'input',
      input: 'connection',
      required: false,
      default: undefined,
      ui_hidden: false,
      type: {
        name: 'AnyField',
        cardinality: 'SINGLE',
        batch: false,
      },
    } satisfies FieldInputTemplate;

    const c = { source, sourceHandle, target, targetHandle: candidate.name };
    return validateConnection(c, nodes, edges, templates, edgePendingUpdate, true) === null ? [candidate] : [];
  }

  const targetTemplate = templates[targetNode.data.type];
  if (!targetTemplate) {
    return [];
  }

  if (!isConnectorNode(sourceNode)) {
    const sourceTemplate = templates[sourceNode.data.type];
    if (!sourceTemplate) {
      return [];
    }

    const sourceField = sourceTemplate.outputs[sourceHandle];

    if (!sourceField) {
      return [];
    }
  }

  const targetCandidateFields = map(targetTemplate.inputs).filter((field) => {
    const c = { source, sourceHandle, target, targetHandle: field.name };
    const connectionErrorTKey = validateConnection(c, nodes, edges, templates, edgePendingUpdate, true);
    return connectionErrorTKey === null;
  });

  return targetCandidateFields;
};

export const getSourceCandidateFields = (
  target: string,
  targetHandle: string,
  source: string,
  nodes: AnyNode[],
  edges: AnyEdge[],
  templates: Templates,
  edgePendingUpdate: AnyEdge | null
): FieldOutputTemplate[] => {
  const targetNode = nodes.find((n) => n.id === target);
  const sourceNode = nodes.find((n) => n.id === source);
  if (!sourceNode || !targetNode) {
    return [];
  }

  if (isConnectorNode(sourceNode)) {
    const sourceFieldType = resolveConnectorSourceFieldType(sourceNode.id, nodes, edges, templates);
    const targetTemplate = !isConnectorNode(targetNode) ? templates[targetNode.data.type] : null;
    const targetFieldType = targetTemplate?.inputs[targetHandle]?.type;
    const candidateType = sourceFieldType ?? targetFieldType;
    if (!candidateType) {
      return [];
    }

    const candidate = {
      name: CONNECTOR_OUTPUT_HANDLE,
      title: 'Connector Output',
      description: '',
      fieldKind: 'output',
      ui_hidden: false,
      type: candidateType,
    } satisfies FieldOutputTemplate;

    const c = { source, sourceHandle: candidate.name, target, targetHandle };
    return validateConnection(c, nodes, edges, templates, edgePendingUpdate, true) === null ? [candidate] : [];
  }

  const sourceTemplate = templates[sourceNode.data.type];
  if (!sourceTemplate) {
    return [];
  }

  if (!isConnectorNode(targetNode)) {
    const targetTemplate = templates[targetNode.data.type];
    if (!targetTemplate) {
      return [];
    }

    const targetField = targetTemplate.inputs[targetHandle];

    if (!targetField) {
      return [];
    }
  } else if (targetHandle !== CONNECTOR_INPUT_HANDLE) {
    return [];
  }

  const sourceCandidateFields = map(sourceTemplate.outputs).filter((field) => {
    const c = { source, sourceHandle: field.name, target, targetHandle };
    const connectionErrorTKey = validateConnection(c, nodes, edges, templates, edgePendingUpdate, true);
    return connectionErrorTKey === null;
  });

  return sourceCandidateFields;
};
