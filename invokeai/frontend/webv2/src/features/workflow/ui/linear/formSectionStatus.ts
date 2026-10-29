import type { InvocationTemplates, ProjectGraphState, WorkflowFormElement } from '@features/workflow/contracts';
import type { ResolvedWorkflowEdge } from '@features/workflow/utility';

import { isInvocationNode } from '@features/workflow/contracts';
import {
  applyNodeFieldFormSettings,
  getFormChildren,
  getNodeFieldFormSettings,
  getResolvedWorkflowEdges,
  getWorkflowFieldInvalidReason,
} from '@features/workflow/utility';

/** Reuse the same field/template validation as NodeFieldControl, including connected inputs and authored bounds. */
export const getInvalidFormFieldIds = (
  projectGraph: ProjectGraphState,
  templates: InvocationTemplates
): Set<string> => {
  const invalid = new Set<string>();
  const nodes = new Map(projectGraph.nodes.filter(isInvocationNode).map((node) => [node.id, node]));
  const edges = getResolvedWorkflowEdges(projectGraph.nodes, projectGraph.edges);

  for (const element of Object.values(projectGraph.form.elements)) {
    if (element.type !== 'node-field') {
      continue;
    }
    if (isFormFieldInvalid(element, nodes, edges, templates)) {
      invalid.add(element.id);
    }
  }
  return invalid;
};

export const sectionHasInvalidField = (
  elements: readonly WorkflowFormElement[],
  projectGraph: ProjectGraphState,
  invalidIds: ReadonlySet<string>
): boolean =>
  elements.some((element) => {
    if (element.type === 'node-field') {
      return invalidIds.has(element.id);
    }
    return element.type === 'container'
      ? sectionHasInvalidField(getFormChildren(projectGraph.form, element.id), projectGraph, invalidIds)
      : false;
  });

const isFormFieldInvalid = (
  element: Extract<WorkflowFormElement, { type: 'node-field' }>,
  nodes: Map<string, Extract<ProjectGraphState['nodes'][number], { type: 'invocation' }>>,
  edges: readonly ResolvedWorkflowEdge[],
  templates: InvocationTemplates
): boolean => {
  const { nodeId, fieldName } = element.data.fieldIdentifier;
  const node = nodes.get(nodeId);
  if (!node) {
    return true;
  }
  const template = node.data.dynamicInputTemplates?.[fieldName] ?? templates[node.data.type]?.inputs[fieldName];
  if (!template) {
    return true;
  }
  const formSettings = getNodeFieldFormSettings(element.data.settings, template);
  const effectiveTemplate = applyNodeFieldFormSettings(template, formSettings);
  const isConnected = edges.some((edge) => edge.target === nodeId && edge.targetHandle === fieldName);
  return (
    getWorkflowFieldInvalidReason({
      isConnected,
      template: effectiveTemplate,
      value: node.data.inputs[fieldName]?.value,
    }) !== null
  );
};
