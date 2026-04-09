import { addElement, getIsFormEmpty } from 'features/nodes/components/sidePanel/builder/form-manipulation';
import { CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX } from 'features/nodes/store/nodesSlice';
import type { Templates } from 'features/nodes/store/types';
import { validateConnectionTypes } from 'features/nodes/store/util/validateConnectionTypes';
import type { FieldInputTemplate, StatefulFieldValue } from 'features/nodes/types/field';
import { isStatefulFieldType } from 'features/nodes/types/field';
import type { AnyEdge, AnyNode } from 'features/nodes/types/invocation';
import { isInvocationNode } from 'features/nodes/types/invocation';
import {
  type BuilderForm,
  buildNodeFieldElement,
  type FormElement,
  getDefaultForm,
  isContainerElement,
  type NodeFieldElement,
  validateFormStructure,
} from 'features/nodes/types/workflow';
import type { paths } from 'services/api/schema';

type WorkflowResponse =
  paths['/api/v1/workflows/i/{workflow_id}']['get']['responses']['200']['content']['application/json'];

type WorkflowNodeLike = {
  data?: {
    id?: string;
    type?: string;
    inputs?: Record<string, { label?: string; description?: string; value?: StatefulFieldValue }>;
  };
};

type ExposedFieldLike = {
  nodeId?: string;
  fieldName?: string;
};

type SavedWorkflowFormFieldData = {
  label: string;
  description: string;
  typeName: string | null;
  isMissing: boolean;
};

type SavedWorkflowDynamicField = {
  fieldName: string;
  fieldTemplate: FieldInputTemplate;
  label: string;
  description: string;
  initialValue: StatefulFieldValue;
  settings: NodeFieldElement['data']['settings'];
};

const getStoredForm = (workflow: WorkflowResponse | undefined): BuilderForm | null => {
  const form = workflow?.workflow.form;

  if (!form || typeof form !== 'object' || !('elements' in form) || !('rootElementId' in form)) {
    return null;
  }

  return form as unknown as BuilderForm;
};

const getWorkflowNodes = (workflow: WorkflowResponse | undefined): WorkflowNodeLike[] => {
  return (workflow?.workflow.nodes ?? []) as WorkflowNodeLike[];
};

const buildFormFromExposedFields = (
  workflow: WorkflowResponse | undefined,
  templates: Templates
): BuilderForm | null => {
  const exposedFields = (workflow?.workflow.exposedFields ?? []) as ExposedFieldLike[];

  if (exposedFields.length === 0) {
    return null;
  }

  const nodes = getWorkflowNodes(workflow);
  const form = getDefaultForm();

  for (const { nodeId, fieldName } of [...exposedFields].reverse()) {
    if (!nodeId || !fieldName) {
      continue;
    }

    const node = nodes.find((candidate) => candidate.data?.id === nodeId);
    const nodeType = node?.data?.type;
    if (!nodeType) {
      continue;
    }

    const fieldTemplate = templates[nodeType]?.inputs[fieldName];
    if (!fieldTemplate) {
      continue;
    }

    const element = buildNodeFieldElement(nodeId, fieldName, fieldTemplate.type);
    element.data.showDescription = false;
    addElement({
      form,
      element,
      parentId: form.rootElementId,
      index: 0,
    });
  }

  return form;
};

export const getRenderableWorkflowForm = (
  workflow: WorkflowResponse | undefined,
  templates: Templates
): BuilderForm | null => {
  const storedForm = getStoredForm(workflow);

  if (storedForm && validateFormStructure(storedForm) && !getIsFormEmpty(storedForm)) {
    return storedForm;
  }

  const fallbackForm = buildFormFromExposedFields(workflow, templates);
  if (fallbackForm && !getIsFormEmpty(fallbackForm)) {
    return fallbackForm;
  }

  if (storedForm && validateFormStructure(storedForm)) {
    return storedForm;
  }

  return null;
};

export const getSavedWorkflowFormFieldData = (
  workflow: WorkflowResponse | undefined,
  templates: Templates,
  element: NodeFieldElement
): SavedWorkflowFormFieldData => {
  const { nodeId, fieldName } = element.data.fieldIdentifier;
  const node = getWorkflowNodes(workflow).find((candidate) => candidate.data?.id === nodeId);
  const nodeType = node?.data?.type;
  const field = node?.data?.inputs?.[fieldName];
  const fieldTemplate = nodeType ? templates[nodeType]?.inputs[fieldName] : undefined;

  return {
    label: field?.label || fieldTemplate?.title || fieldName,
    description: fieldTemplate?.description || '',
    typeName: fieldTemplate?.type.name ?? null,
    isMissing: !node || !field || !fieldTemplate,
  };
};

const getElementsInOrder = (form: BuilderForm): FormElement[] => {
  const orderedElements: FormElement[] = [];

  const visit = (elementId: string) => {
    const element = form.elements[elementId];
    if (!element) {
      return;
    }

    orderedElements.push(element);
    if (isContainerElement(element)) {
      for (const childId of element.data.children) {
        visit(childId);
      }
    }
  };

  visit(form.rootElementId);
  return orderedElements;
};

const buildDynamicFieldName = (nodeId: string, fieldName: string): string => {
  return `${CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX}${nodeId}::${fieldName}`;
};

const cloneDynamicFieldTemplate = ({
  fieldName,
  fieldTemplate,
  label,
  description,
  uiOrder,
}: {
  fieldName: string;
  fieldTemplate: FieldInputTemplate;
  label: string;
  description: string;
  uiOrder: number;
}): FieldInputTemplate => {
  return {
    ...fieldTemplate,
    name: fieldName,
    title: label,
    description,
    ui_order: uiOrder,
    input: 'any',
    ui_hidden: false,
  } as FieldInputTemplate;
};

export const getSavedWorkflowDynamicFields = (
  workflow: WorkflowResponse | undefined,
  templates: Templates
): SavedWorkflowDynamicField[] => {
  const form = getRenderableWorkflowForm(workflow, templates);
  if (!form) {
    return [];
  }

  const nodes = getWorkflowNodes(workflow);
  const dynamicFields: SavedWorkflowDynamicField[] = [];

  for (const element of getElementsInOrder(form)) {
    if (!('data' in element) || !('fieldIdentifier' in (element.data ?? {}))) {
      continue;
    }
    if (!('type' in element) || element.type !== 'node-field') {
      continue;
    }

    const { nodeId, fieldName } = element.data.fieldIdentifier;
    const node = nodes.find((candidate) => candidate.data?.id === nodeId);
    const nodeType = node?.data?.type;
    const field = node?.data?.inputs?.[fieldName];
    const fieldTemplate = nodeType ? templates[nodeType]?.inputs[fieldName] : undefined;

    if (!field || !fieldTemplate || !isStatefulFieldType(fieldTemplate.type)) {
      continue;
    }

    const dynamicFieldName = buildDynamicFieldName(nodeId, fieldName);
    const label = field.label || fieldTemplate.title || fieldName;
    const description = field.description || fieldTemplate.description || '';

    dynamicFields.push({
      fieldName: dynamicFieldName,
      fieldTemplate: cloneDynamicFieldTemplate({
        fieldName: dynamicFieldName,
        fieldTemplate,
        label,
        description,
        uiOrder: dynamicFields.length,
      }),
      label,
      description,
      initialValue: field.value,
      settings: element.data.settings,
    });
  }

  return dynamicFields;
};

export const getSavedWorkflowDynamicEdgeIdsToRemove = ({
  nodeId,
  fields,
  nodes,
  edges,
  templates,
}: {
  nodeId: string;
  fields: SavedWorkflowDynamicField[];
  nodes: AnyNode[];
  edges: AnyEdge[];
  templates: Templates;
}): string[] => {
  const nextFieldTemplates = new Map(fields.map((field) => [field.fieldName, field.fieldTemplate]));

  return edges.flatMap((edge) => {
    if (edge.type !== 'default' || edge.target !== nodeId || !edge.targetHandle) {
      return [];
    }

    if (!edge.targetHandle.startsWith(CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX)) {
      return [];
    }

    const targetFieldTemplate = nextFieldTemplates.get(edge.targetHandle);
    if (!targetFieldTemplate || targetFieldTemplate.input === 'direct' || !edge.sourceHandle) {
      return [edge.id];
    }

    const sourceNode = nodes.find((node) => node.id === edge.source);
    if (!sourceNode || !isInvocationNode(sourceNode)) {
      return [edge.id];
    }

    const sourceTemplate = templates[sourceNode.data.type];
    const sourceFieldTemplate = sourceTemplate?.outputs[edge.sourceHandle];
    if (!sourceFieldTemplate) {
      return [edge.id];
    }

    if (!validateConnectionTypes(sourceFieldTemplate.type, targetFieldTemplate.type)) {
      return [edge.id];
    }

    return [];
  });
};
