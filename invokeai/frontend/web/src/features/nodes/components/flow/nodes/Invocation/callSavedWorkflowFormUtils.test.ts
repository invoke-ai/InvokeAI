import { buildEdge, buildNode, call_saved_workflow, templates } from 'features/nodes/store/util/testUtils';
import type { BooleanFieldInputTemplate } from 'features/nodes/types/field';
import {
  type BuilderForm,
  buildHeading,
  buildNodeFieldElement,
  getDefaultForm,
  isContainerElement,
} from 'features/nodes/types/workflow';
import type { paths } from 'services/api/schema';
import { describe, expect, it } from 'vitest';

import {
  getRenderableWorkflowForm,
  getSavedWorkflowDynamicEdgeIdsToRemove,
  getSavedWorkflowDynamicFields,
  getSavedWorkflowFormFieldData,
  shouldSyncSavedWorkflowDynamicFields,
} from './callSavedWorkflowFormUtils';

type WorkflowResponse =
  paths['/api/v1/workflows/i/{workflow_id}']['get']['responses']['200']['content']['application/json'];

const addTemplate = templates.add;
if (!addTemplate) {
  throw new Error('Expected add template');
}
const addInputA = addTemplate.inputs.a;
const addInputB = addTemplate.inputs.b;
if (!addInputA || !addInputB) {
  throw new Error('Expected add template inputs');
}

const getRootChildren = (form: BuilderForm): string[] => {
  const root = form.elements[form.rootElementId];

  if (!root || !isContainerElement(root)) {
    throw new Error('Expected root container');
  }

  return root.data.children;
};

const buildWorkflowResponse = (overrides?: {
  exposedFields?: Array<{ nodeId: string; fieldName: string }>;
  form?: BuilderForm | null;
  inputs?: Record<string, { name: string; label?: string; value?: unknown }>;
}): WorkflowResponse =>
  ({
    workflow_id: 'workflow-1',
    name: 'Workflow 1',
    created_at: '2026-04-08T00:00:00Z',
    updated_at: '2026-04-08T00:00:00Z',
    opened_at: null,
    user_id: 'user-1',
    is_public: false,
    thumbnail_url: null,
    workflow: {
      id: 'workflow-1',
      name: 'Workflow 1',
      author: 'InvokeAI',
      description: 'A workflow',
      version: '1.0.0',
      contact: '',
      tags: '',
      notes: '',
      exposedFields: overrides?.exposedFields ?? [],
      meta: {
        category: 'user',
        version: '3.0.0',
      },
      nodes: [
        {
          id: 'node-1',
          type: 'invocation',
          data: {
            id: 'node-1',
            type: 'add',
            inputs: overrides?.inputs ?? {
              a: { name: 'a', label: 'Left Addend', value: 1 },
              b: { name: 'b', label: '', value: 2 },
            },
          },
        },
      ],
      edges: [],
      form: overrides?.form ?? getDefaultForm(),
    },
  }) as WorkflowResponse;

describe('callSavedWorkflowFormUtils', () => {
  it('does not sync dynamic fields while a selected saved workflow is still loading', () => {
    expect(
      shouldSyncSavedWorkflowDynamicFields({
        workflowId: 'workflow-1',
        workflow: undefined,
      })
    ).toBe(false);
  });

  it('syncs dynamic fields when no workflow is selected or when the selected workflow is loaded', () => {
    expect(
      shouldSyncSavedWorkflowDynamicFields({
        workflowId: '',
        workflow: undefined,
      })
    ).toBe(true);
    expect(
      shouldSyncSavedWorkflowDynamicFields({
        workflowId: 'workflow-1',
        workflow: buildWorkflowResponse(),
      })
    ).toBe(true);
  });

  it('returns the stored form when it is non-empty and valid', () => {
    const form = getDefaultForm();
    const heading = buildHeading('Workflow Inputs');
    form.elements[heading.id] = { ...heading, parentId: form.rootElementId };
    getRootChildren(form).push(heading.id);

    const workflow = buildWorkflowResponse({ form });

    expect(getRenderableWorkflowForm(workflow, templates)).toBe(form);
  });

  it('builds a fallback form from exposed fields when the stored form is empty', () => {
    const workflow = buildWorkflowResponse({
      exposedFields: [{ nodeId: 'node-1', fieldName: 'a' }],
    });

    const form = getRenderableWorkflowForm(workflow, templates);

    expect(form).not.toBeNull();
    expect(form ? getRootChildren(form) : []).toHaveLength(1);
    const childId = form ? getRootChildren(form)[0] : undefined;
    expect(childId).toBeDefined();
    expect(childId ? form?.elements[childId]?.type : undefined).toBe('node-field');
  });

  it('skips exposed fields that do not resolve to a known node field', () => {
    const workflow = buildWorkflowResponse({
      exposedFields: [{ nodeId: 'missing-node', fieldName: 'a' }],
    });

    const form = getRenderableWorkflowForm(workflow, templates);

    expect(form).not.toBeNull();
    expect(form ? getRootChildren(form) : []).toHaveLength(0);
  });

  it('uses the stored field label when available', () => {
    const element = buildNodeFieldElement('node-1', 'a', addInputA.type);
    const workflow = buildWorkflowResponse();

    expect(getSavedWorkflowFormFieldData(workflow, templates, element)).toEqual(
      expect.objectContaining({
        label: 'Left Addend',
        description: 'The first number',
        typeName: 'IntegerField',
        isMissing: false,
      })
    );
  });

  it('falls back to the template title when the stored field label is empty', () => {
    const element = buildNodeFieldElement('node-1', 'b', addInputB.type);
    const workflow = buildWorkflowResponse();

    expect(getSavedWorkflowFormFieldData(workflow, templates, element)).toEqual(
      expect.objectContaining({
        label: 'B',
        description: 'The second number',
        typeName: 'IntegerField',
        isMissing: false,
      })
    );
  });

  it('marks missing node field references as missing', () => {
    const element = buildNodeFieldElement('missing-node', 'a', addInputA.type);
    const workflow = buildWorkflowResponse();

    expect(getSavedWorkflowFormFieldData(workflow, templates, element)).toEqual(
      expect.objectContaining({
        label: 'a',
        description: '',
        typeName: null,
        isMissing: true,
      })
    );
  });

  it('builds ordered dynamic fields from the workflow form', () => {
    const workflow = buildWorkflowResponse({
      exposedFields: [
        { nodeId: 'node-1', fieldName: 'a' },
        { nodeId: 'node-1', fieldName: 'b' },
      ],
    });

    const dynamicFields = getSavedWorkflowDynamicFields(workflow, templates);

    expect(dynamicFields).toHaveLength(2);
    expect(dynamicFields[0]?.fieldName).toBe('saved_workflow_input::node-1::a');
    expect(dynamicFields[1]?.fieldName).toBe('saved_workflow_input::node-1::b');
    expect(dynamicFields[0]?.fieldTemplate.title).toBe('Left Addend');
    expect(dynamicFields[1]?.fieldTemplate.title).toBe('B');
    expect(dynamicFields[0]?.fieldTemplate.name).toBe(dynamicFields[0]?.fieldName);
    expect(dynamicFields[1]?.fieldTemplate.ui_order).toBe(1);
    expect(dynamicFields[0]?.initialValue).toBe(1);
    expect(dynamicFields[1]?.initialValue).toBe(2);
  });

  it('preserves inbound edges when the same exposed dynamic field remains compatible', () => {
    const sourceNode = buildNode(addTemplate);
    const targetNode = buildNode(call_saved_workflow);
    const fields = getSavedWorkflowDynamicFields(
      buildWorkflowResponse({
        exposedFields: [{ nodeId: 'node-1', fieldName: 'a' }],
      }),
      templates
    );

    const edgeIdsToRemove = getSavedWorkflowDynamicEdgeIdsToRemove({
      nodeId: targetNode.id,
      fields,
      nodes: [sourceNode, targetNode],
      edges: [buildEdge(sourceNode.id, 'value', targetNode.id, 'saved_workflow_input::node-1::a')],
      templates,
    });

    expect(edgeIdsToRemove).toEqual([]);
  });

  it('removes inbound edges when a previously connected field is no longer exposed', () => {
    const sourceNode = buildNode(addTemplate);
    const targetNode = buildNode(call_saved_workflow);
    const edge = buildEdge(sourceNode.id, 'value', targetNode.id, 'saved_workflow_input::node-1::a');

    const edgeIdsToRemove = getSavedWorkflowDynamicEdgeIdsToRemove({
      nodeId: targetNode.id,
      fields: [],
      nodes: [sourceNode, targetNode],
      edges: [edge],
      templates,
    });

    expect(edgeIdsToRemove).toEqual([edge.id]);
  });

  it('removes inbound edges when an exposed field changes to an incompatible type', () => {
    const sourceNode = buildNode(addTemplate);
    const targetNode = buildNode(call_saved_workflow);
    const edge = buildEdge(sourceNode.id, 'value', targetNode.id, 'saved_workflow_input::node-1::a');

    const booleanTemplate: BooleanFieldInputTemplate = {
      name: 'saved_workflow_input::node-1::a',
      title: 'Enabled',
      required: false,
      description: 'Whether the workflow is enabled',
      fieldKind: 'input',
      input: 'any',
      ui_hidden: false,
      default: false,
      type: {
        name: 'BooleanField',
        cardinality: 'SINGLE',
        batch: false,
      },
    };

    const edgeIdsToRemove = getSavedWorkflowDynamicEdgeIdsToRemove({
      nodeId: targetNode.id,
      fields: [
        {
          fieldName: 'saved_workflow_input::node-1::a',
          fieldTemplate: booleanTemplate,
          label: 'Enabled',
          description: 'Whether the workflow is enabled',
          initialValue: false,
          settings: undefined,
        },
      ],
      nodes: [sourceNode, targetNode],
      edges: [edge],
      templates,
    });

    expect(edgeIdsToRemove).toEqual([edge.id]);
  });
});
