import { call_saved_workflows, buildNode } from 'features/nodes/store/util/testUtils';
import { getInitialWorkflow } from 'features/nodes/store/nodesSlice';
import { describe, expect, it } from 'vitest';

describe('buildWorkflowFast', () => {
  it('persists the selected workflow id for call_saved_workflows nodes', async () => {
    Object.assign(globalThis, {
      window: {
        location: {
          origin: 'http://localhost',
        },
      },
    });

    const { buildWorkflowFast } = await import('features/nodes/util/workflow/buildWorkflow');
    const node = buildNode(call_saved_workflows);
    node.data.inputs.workflow_id.value = 'workflow-123';

    const workflow = buildWorkflowFast({
      _version: 1,
      formFieldInitialValues: {},
      ...getInitialWorkflow(),
      nodes: [node],
      edges: [],
    });

    expect(workflow.nodes).toHaveLength(1);
    expect(workflow.nodes[0]?.type).toBe('invocation');
    if (workflow.nodes[0]?.type !== 'invocation') {
      throw new Error('Expected invocation node');
    }
    expect(workflow.nodes[0].data.type).toBe('call_saved_workflows');
    expect(workflow.nodes[0].data.inputs.workflow_id.value).toBe('workflow-123');
  });
});
