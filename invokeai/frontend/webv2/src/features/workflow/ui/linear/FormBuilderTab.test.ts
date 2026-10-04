import type { FieldInputTemplate, InvocationTemplate } from '@features/workflow/contracts';

import { buildInvocationNode, createProjectGraph } from '@features/workflow/utility';
import { describe, expect, it } from 'vitest';

import { getInvalidNodeFieldElementIds } from './FormBuilderTab';

const dynamicInput: FieldInputTemplate = {
  default: '',
  description: 'Dynamic input',
  exclusiveMaximum: null,
  exclusiveMinimum: null,
  fieldKind: 'input',
  input: 'any',
  maximum: null,
  minimum: null,
  multipleOf: null,
  name: 'dynamic_input',
  options: null,
  required: true,
  title: 'Dynamic input',
  type: { batch: false, cardinality: 'SINGLE', name: 'StringField' },
  uiChoiceLabels: null,
  uiComponent: null,
  uiHidden: false,
  uiModelBase: null,
  uiModelFormat: null,
  uiModelType: null,
  uiOrder: null,
};

const callSavedWorkflowTemplate: InvocationTemplate = {
  category: 'workflow',
  classification: 'stable',
  description: '',
  inputs: {},
  nodePack: 'invokeai',
  outputs: {},
  outputType: 'call_saved_workflow_output',
  tags: [],
  title: 'Call Saved Workflow',
  type: 'call_saved_workflow',
  useCache: true,
  version: '1.0.0',
};

describe('getInvalidNodeFieldElementIds', () => {
  it('validates a form field through its persisted dynamic input template', () => {
    const node = buildInvocationNode(callSavedWorkflowTemplate, { x: 0, y: 0 });
    node.id = 'call';
    node.data.dynamicInputTemplates = { dynamic_input: dynamicInput };
    node.data.inputs.dynamic_input = { label: 'Dynamic input', name: 'dynamic_input', value: 'ready' };
    const document = createProjectGraph('form-builder-dynamic-input');
    document.nodes = [node];
    const root = document.form.elements[document.form.rootElementId];

    if (root?.type !== 'container') {
      throw new Error('Expected a form root container.');
    }

    root.data.children.push('dynamic-field');
    document.form.elements['dynamic-field'] = {
      data: {
        fieldIdentifier: { fieldName: 'dynamic_input', nodeId: node.id },
        showDescription: false,
        showShuffle: false,
      },
      id: 'dynamic-field',
      parentId: root.id,
      type: 'node-field',
    };

    expect(
      getInvalidNodeFieldElementIds(document, 'loaded', { call_saved_workflow: callSavedWorkflowTemplate })
    ).toEqual(new Set());
  });
});
