import { buildNode, templates } from 'features/nodes/store/util/testUtils';
import type { IntegerFieldInputTemplate } from 'features/nodes/types/field';
import { describe, expect, it } from 'vitest';

import { callSavedWorkflowDynamicFieldsChanged, fieldIntegerValueChanged, nodesSliceConfig } from './nodesSlice';

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
  input: 'any' as const,
});

describe('callSavedWorkflowDynamicFieldsChanged', () => {
  it('seeds new dynamic fields with the source workflow values', () => {
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
      })
    );

    const dynamicField = nextState.nodes[0];
    if (!dynamicField || dynamicField.type !== 'invocation') {
      throw new Error('Expected invocation node');
    }

    expect(dynamicField.data.inputs['saved_workflow_input::node-1::a']?.value).toBe(23);
    expect(dynamicField.data.inputs['saved_workflow_input::node-1::a']?.label).toBe('Left Addend');
    expect(dynamicField.data.dynamicInputTemplates['saved_workflow_input::node-1::a']?.title).toBe('Left Addend');
  });

  it('preserves existing dynamic field values on resync', () => {
    const state = nodesSliceConfig.getInitialState();
    const node = buildNode(callSavedWorkflowTemplate);
    state.nodes.push(node);

    const fieldName = 'saved_workflow_input::node-1::a';

    let nextState = nodesSliceConfig.slice.reducer(
      state,
      callSavedWorkflowDynamicFieldsChanged({
        nodeId: node.id,
        fields: [
          {
            fieldName,
            fieldTemplate: buildDynamicIntegerTemplate(fieldName),
            label: 'Left Addend',
            description: 'The first number',
            initialValue: 23,
          },
        ],
      })
    );

    nextState = nodesSliceConfig.slice.reducer(
      nextState,
      fieldIntegerValueChanged({
        nodeId: node.id,
        fieldName,
        value: 99,
      })
    );

    nextState = nodesSliceConfig.slice.reducer(
      nextState,
      callSavedWorkflowDynamicFieldsChanged({
        nodeId: node.id,
        fields: [
          {
            fieldName,
            fieldTemplate: buildDynamicIntegerTemplate(fieldName),
            label: 'Left Addend',
            description: 'The first number',
            initialValue: 23,
          },
        ],
      })
    );

    const resyncedNode = nextState.nodes[0];
    if (!resyncedNode || resyncedNode.type !== 'invocation') {
      throw new Error('Expected invocation node');
    }

    expect(resyncedNode.data.inputs[fieldName]?.value).toBe(99);
    expect(resyncedNode.data.dynamicInputTemplates[fieldName]?.name).toBe(fieldName);
  });

  it('removes stale dynamic field templates when the selected workflow fields change', () => {
    const state = nodesSliceConfig.getInitialState();
    const node = buildNode(callSavedWorkflowTemplate);
    state.nodes.push(node);

    const fieldName = 'saved_workflow_input::node-1::a';

    let nextState = nodesSliceConfig.slice.reducer(
      state,
      callSavedWorkflowDynamicFieldsChanged({
        nodeId: node.id,
        fields: [
          {
            fieldName,
            fieldTemplate: buildDynamicIntegerTemplate(fieldName),
            label: 'Left Addend',
            description: 'The first number',
            initialValue: 23,
          },
        ],
      })
    );

    nextState = nodesSliceConfig.slice.reducer(
      nextState,
      callSavedWorkflowDynamicFieldsChanged({
        nodeId: node.id,
        fields: [],
      })
    );

    const resyncedNode = nextState.nodes[0];
    if (!resyncedNode || resyncedNode.type !== 'invocation') {
      throw new Error('Expected invocation node');
    }

    expect(resyncedNode.data.inputs[fieldName]).toBeUndefined();
    expect(resyncedNode.data.dynamicInputTemplates[fieldName]).toBeUndefined();
  });
});
