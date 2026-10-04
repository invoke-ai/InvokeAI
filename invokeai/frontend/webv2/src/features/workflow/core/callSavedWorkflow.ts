import type {
  FieldInputTemplate,
  InvocationTemplates,
  ProjectGraphState,
  WorkflowForm,
  WorkflowFormElement,
  WorkflowInvocationNode,
} from './types';

import { getResolvedWorkflowEdges } from './connectors';
import { getEffectiveWorkflowFieldDescription, isDirectInputField, isWorkflowFieldValueValid } from './fields';
import { isInvocationNode } from './types';
import { validateConnectionTypes } from './validation';

/** Prefix used by the backend to distinguish child-workflow inputs from node inputs. */
export const CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX = 'saved_workflow_input::';

export type CallSavedWorkflowStatus = 'loading' | 'ready' | 'error';

export interface SavedWorkflowDynamicField {
  fieldName: string;
  fieldTemplate: FieldInputTemplate;
  label: string;
  description: string;
  initialValue: unknown;
  settings?: Record<string, unknown>;
}

const getFormElementsInOrder = (form: WorkflowForm): WorkflowFormElement[] => {
  const ordered: WorkflowFormElement[] = [];
  const visited = new Set<string>();

  const visit = (elementId: string): void => {
    if (visited.has(elementId)) {
      return;
    }

    const element = form.elements[elementId];

    if (!element) {
      return;
    }

    visited.add(elementId);
    ordered.push(element);

    if (element.type === 'container') {
      element.data.children.forEach(visit);
    }
  };

  visit(form.rootElementId);
  return ordered;
};

const getDynamicFieldName = (nodeId: string, fieldName: string): string =>
  `${CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX}${nodeId}::${fieldName}`;

const cloneDynamicFieldTemplate = (
  fieldName: string,
  template: FieldInputTemplate,
  label: string,
  description: string
): FieldInputTemplate => ({
  ...template,
  description,
  input: 'any',
  name: fieldName,
  title: label,
  uiHidden: false,
  uiOrder: null,
});

/** Returns the fields exposed by a selected workflow in the order shown by its form. */
export const getSavedWorkflowDynamicFields = (
  workflow: ProjectGraphState | undefined,
  templates: InvocationTemplates
): SavedWorkflowDynamicField[] => {
  if (!workflow) {
    return [];
  }

  const dynamicFields: SavedWorkflowDynamicField[] = [];
  const seenFieldNames = new Set<string>();

  for (const element of getFormElementsInOrder(workflow.form)) {
    if (element.type !== 'node-field') {
      continue;
    }

    const { fieldName, nodeId } = element.data.fieldIdentifier;
    const node = workflow.nodes.find((candidate) => candidate.id === nodeId);

    if (!node || !isInvocationNode(node)) {
      continue;
    }

    const field = node.data.inputs[fieldName];
    const fieldTemplate = templates[node.data.type]?.inputs[fieldName];

    if (!field || !fieldTemplate || !isDirectInputField(fieldTemplate)) {
      continue;
    }

    const dynamicFieldName = getDynamicFieldName(nodeId, fieldName);

    if (seenFieldNames.has(dynamicFieldName)) {
      continue;
    }

    seenFieldNames.add(dynamicFieldName);
    const label = field.label || fieldTemplate.title || fieldName;
    const description = getEffectiveWorkflowFieldDescription(field, fieldTemplate);

    dynamicFields.push({
      description,
      fieldName: dynamicFieldName,
      fieldTemplate: cloneDynamicFieldTemplate(dynamicFieldName, fieldTemplate, label, description),
      initialValue: field.value,
      label,
      settings: element.data.settings,
    });
  }

  return dynamicFields;
};

/** Selects query data only when it belongs to the currently selected workflow id. */
export const getSelectedSavedWorkflow = <T extends { workflow_id: string }>(
  workflowId: string | null | undefined,
  workflow: T | undefined
): T | undefined => (workflowId && workflow?.workflow_id === workflowId ? workflow : undefined);

export const shouldSyncSavedWorkflowDynamicFields = ({
  workflowId,
  workflow,
}: {
  workflowId: string | null | undefined;
  workflow: ProjectGraphState | undefined;
}): boolean => !workflowId || workflow !== undefined;

/** Returns dynamic inbound edges that no longer have a compatible target field. */
export const getSavedWorkflowDynamicEdgeIdsToRemove = (
  document: Pick<ProjectGraphState, 'edges' | 'nodes'>,
  nodeId: string,
  fields: SavedWorkflowDynamicField[],
  templates: InvocationTemplates
): string[] => {
  const nextFieldTemplates = new Map(fields.map((field) => [field.fieldName, field.fieldTemplate]));
  const resolvedEdges = getResolvedWorkflowEdges(document.nodes, document.edges, templates);

  return resolvedEdges.flatMap((edge) => {
    if (
      edge.type !== 'default' ||
      edge.target !== nodeId ||
      !edge.targetHandle.startsWith(CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX)
    ) {
      return [];
    }

    const targetTemplate = nextFieldTemplates.get(edge.targetHandle);
    const sourceNode = document.nodes.find((node) => node.id === edge.source);
    const sourceTemplate = sourceNode && isInvocationNode(sourceNode) ? templates[sourceNode.data.type] : undefined;
    const sourceField = sourceTemplate?.outputs[edge.sourceHandle];

    return targetTemplate && targetTemplate.input !== 'direct' && sourceField
      ? validateConnectionTypes(sourceField.type, targetTemplate.type)
        ? []
        : [edge.id]
      : [edge.id];
  });
};

const sameFieldType = (left: FieldInputTemplate, right: FieldInputTemplate): boolean =>
  left.type.name === right.type.name &&
  left.type.cardinality === right.type.cardinality &&
  left.type.batch === right.type.batch;

const removeFormElement = (form: WorkflowForm, elementId: string): WorkflowForm => {
  const removed = form.elements[elementId];

  if (!removed || elementId === form.rootElementId) {
    return form;
  }

  const elements = Object.fromEntries(Object.entries(form.elements).filter(([id]) => id !== elementId));

  for (const [id, element] of Object.entries(elements)) {
    if (element.type === 'container') {
      elements[id] = {
        ...element,
        data: { ...element.data, children: element.data.children.filter((childId) => childId !== elementId) },
      };
    }
  }

  return { ...form, elements };
};

/**
 * Applies the selected child workflow's field signature to a call node.
 * Existing values survive compatible refreshes; incompatible or new fields use
 * the child workflow's initial values.
 */
export const syncCallSavedWorkflowFields = (
  document: ProjectGraphState,
  nodeId: string,
  fields: SavedWorkflowDynamicField[],
  edgeIdsToRemove: string[],
  status: CallSavedWorkflowStatus = 'ready'
): ProjectGraphState => {
  const node = document.nodes.find((candidate) => candidate.id === nodeId);

  if (!node || !isInvocationNode(node) || node.data.type !== 'call_saved_workflow') {
    return document;
  }

  const uniqueFields = fields.filter(
    (field, index) => fields.findIndex((candidate) => candidate.fieldName === field.fieldName) === index
  );
  const nextFieldNames = new Set(uniqueFields.map((field) => field.fieldName));
  const previousTemplates = node.data.dynamicInputTemplates ?? {};
  const nextInputs: typeof node.data.inputs = {};
  const nextTemplates: NonNullable<typeof node.data.dynamicInputTemplates> = {};

  for (const [name, instance] of Object.entries(node.data.inputs)) {
    if (!name.startsWith(CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX)) {
      nextInputs[name] = instance;
    }
  }

  for (const field of uniqueFields) {
    const previous = node.data.inputs[field.fieldName];
    const previousTemplate = previousTemplates[field.fieldName];
    const keepValue =
      previous !== undefined &&
      (!previousTemplate || sameFieldType(previousTemplate, field.fieldTemplate)) &&
      isWorkflowFieldValueValid(field.fieldTemplate, previous.value);
    const labelOverridden =
      previous &&
      (previous.labelOverride === true ||
        (previousTemplate !== undefined && previous.label !== previousTemplate.title));
    const descriptionOverridden =
      previous &&
      (previous.descriptionOverride === true ||
        (previous.descriptionOverride === undefined &&
          (previousTemplate !== undefined
            ? (previous.description ?? '') !== previousTemplate.description
            : (previous.description ?? '') !== '')));
    const label = labelOverridden && previous ? previous.label : field.label;
    const description = descriptionOverridden && previous ? previous.description : field.description;

    nextTemplates[field.fieldName] = field.fieldTemplate;
    nextInputs[field.fieldName] = {
      ...(keepValue ? previous : {}),
      description,
      ...(descriptionOverridden ? { descriptionOverride: true } : {}),
      label,
      ...(labelOverridden ? { labelOverride: true } : {}),
      name: field.fieldName,
      value: keepValue ? previous.value : field.initialValue,
    };
  }

  const nextNode: WorkflowInvocationNode = {
    ...node,
    data: { ...node.data, callSavedWorkflowStatus: status, dynamicInputTemplates: nextTemplates, inputs: nextInputs },
  };

  const removedEdgeIds = new Set(edgeIdsToRemove);
  const nextForm = Object.values(document.form.elements)
    .filter(
      (element) =>
        element.type === 'node-field' &&
        element.data.fieldIdentifier.nodeId === nodeId &&
        element.data.fieldIdentifier.fieldName.startsWith(CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX) &&
        !nextFieldNames.has(element.data.fieldIdentifier.fieldName)
    )
    .reduce((form, element) => removeFormElement(form, element.id), document.form);

  return {
    ...document,
    edges: document.edges.filter((edge) => !removedEdgeIds.has(edge.id)),
    form: nextForm,
    nodes: document.nodes.map((candidate) => (candidate.id === nodeId ? nextNode : candidate)),
  };
};

/** Used when a call node is cleared or retargeted before the next child query resolves. */
export const clearSavedWorkflowDynamicFields = (document: ProjectGraphState, nodeId: string): ProjectGraphState => {
  const node = document.nodes.find((candidate) => candidate.id === nodeId);

  if (!node || !isInvocationNode(node) || node.data.type !== 'call_saved_workflow') {
    return document;
  }

  const edgeIds = document.edges
    .filter((edge) => edge.target === nodeId && edge.targetHandle.startsWith(CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX))
    .map((edge) => edge.id);

  return syncCallSavedWorkflowFields(
    {
      ...document,
      nodes: document.nodes.map((candidate) =>
        candidate.id === nodeId && isInvocationNode(candidate)
          ? { ...candidate, data: { ...candidate.data, dynamicInputTemplates: {} } }
          : candidate
      ),
    },
    nodeId,
    [],
    edgeIds
  );
};

export const setCallSavedWorkflowStatus = (
  document: ProjectGraphState,
  nodeId: string,
  status: CallSavedWorkflowStatus
): ProjectGraphState => {
  const node = document.nodes.find((candidate) => candidate.id === nodeId);

  if (!node || !isInvocationNode(node) || node.data.type !== 'call_saved_workflow') {
    return document;
  }

  if (node.data.callSavedWorkflowStatus === status) {
    return document;
  }

  return {
    ...document,
    nodes: document.nodes.map((candidate) =>
      candidate.id === nodeId && isInvocationNode(candidate)
        ? { ...candidate, data: { ...candidate.data, callSavedWorkflowStatus: status } }
        : candidate
    ),
  };
};
