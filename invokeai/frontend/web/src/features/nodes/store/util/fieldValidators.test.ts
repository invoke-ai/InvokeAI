import { callSavedWorkflowDynamicFieldsChanged, nodesSliceConfig } from 'features/nodes/store/nodesSlice';
import { buildNode, templates } from 'features/nodes/store/util/testUtils';
import type { IntegerFieldInputTemplate } from 'features/nodes/types/field';
import { describe, expect, it } from 'vitest';

import { getInvocationNodeErrors } from './fieldValidators';

const callSavedWorkflowTemplate = templates.call_saved_workflow;
const addTemplate = templates.add;

if (!callSavedWorkflowTemplate || !addTemplate || !addTemplate.inputs.a) {
  throw new Error('Expected saved workflow and add templates');
}

const addIntegerInputTemplate = addTemplate.inputs.a as IntegerFieldInputTemplate;

const buildDynamicIntegerTemplate = (fieldName: string): IntegerFieldInputTemplate => ({
  ...addIntegerInputTemplate,
  name: fieldName,
  title: 'Left Addend',
  input: 'any',
});

describe('getInvocationNodeErrors', () => {
  it('does not report missing field templates for dynamic saved workflow inputs', () => {
    const state = nodesSliceConfig.getInitialState();
    const node = buildNode(callSavedWorkflowTemplate);
    state.nodes.push(node);

    const nextState = nodesSliceConfig.slice.reducer(
      state,
      callSavedWorkflowDynamicFieldsChanged({
        nodeId: node.id,
        fields: [
          {
            fieldName: 'saved_workflow_input::node-1::a',
            fieldTemplate: buildDynamicIntegerTemplate('saved_workflow_input::node-1::a'),
            label: 'Left Addend',
            description: 'The first number',
            initialValue: 23,
          },
        ],
        edgeIdsToRemove: [],
      })
    );

    const errors = getInvocationNodeErrors(node.id, templates, nextState);

    expect(
      errors.find((error) => error.type === 'node-error' && error.issue === 'parameters.invoke.missingFieldTemplate')
    ).toBeUndefined();
  });
});
