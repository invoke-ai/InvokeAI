import { callSavedWorkflowDynamicFieldsChanged, nodesSliceConfig } from 'features/nodes/store/nodesSlice';
import { buildNode, templates } from 'features/nodes/store/util/testUtils';
import type { IntegerFieldInputTemplate } from 'features/nodes/types/field';
import { describe, expect, it } from 'vitest';

import { buildNodesGraph } from './buildNodesGraph';

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

describe('buildNodesGraph', () => {
  it('includes dynamic saved workflow inputs when templates are stored on the node', () => {
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

    const rootState = {
      nodes: {
        past: [],
        future: [],
        present: nextState,
      },
      gallery: {
        autoAddBoardId: 'none',
      },
    } as never;

    const graph = buildNodesGraph(rootState, templates);

    expect(graph.nodes[node.id]).toMatchObject({
      workflow_id: '',
      ['saved_workflow_input::node-1::a']: 23,
    });
  });
});
