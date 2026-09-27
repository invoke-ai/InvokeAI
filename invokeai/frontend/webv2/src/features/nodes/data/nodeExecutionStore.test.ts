import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./transport', () => ({
  browserNodesDataPort: {
    buildUrl: (path: string) => `https://api.test${path}`,
    request: vi.fn(),
    requestJson: vi.fn(),
  },
}));

import { nodeExecutionStore } from './nodeExecutionStore';

beforeEach(() => {
  nodeExecutionStore.clearAll();
});

describe('node execution lifecycle', () => {
  it('clears a previous thumbnail when the current result has no image', () => {
    nodeExecutionStore.completed({ invocation_source_id: 'node-1', result: { image: { image_name: 'old.png' } } });
    nodeExecutionStore.completed({ invocation_source_id: 'node-1', result: { type: 'integer_output', value: 1 } });

    expect(nodeExecutionStore.get('node-1')).toMatchObject({
      latestOutput: { type: 'integer_output', value: 1 },
      outputImageUrl: null,
      status: 'completed',
    });
  });
  it('preserves the latest image across progress and failure transitions', () => {
    nodeExecutionStore.completed({
      invocation_source_id: 'node-1',
      result: { image: { image_name: 'result image.png' } },
    });
    nodeExecutionStore.progress('node-1', 0.5, 'Sampling');
    nodeExecutionStore.failed({ error_message: 'Out of memory', invocation_source_id: 'node-1' });

    expect(nodeExecutionStore.get('node-1')).toEqual({
      error: 'Out of memory',
      outputImageUrl: 'https://api.test/api/v1/images/i/result%20image.png/thumbnail',
      latestOutput: { image: { image_name: 'result image.png' } },
      progress: null,
      progressMessage: null,
      status: 'failed',
    });
  });

  it('keeps only the most recent result of a run', () => {
    nodeExecutionStore.started({ invocation_source_id: 'node-1' });
    nodeExecutionStore.completed({ invocation_source_id: 'node-1', result: { type: 'integer_output', value: 1 } });
    nodeExecutionStore.started({ invocation_source_id: 'node-1' });
    nodeExecutionStore.completed({ invocation_source_id: 'node-1', result: { type: 'integer_output', value: 2 } });

    expect(nodeExecutionStore.get('node-1')?.latestOutput).toEqual({ type: 'integer_output', value: 2 });
  });

  it('shows an image returned through workflow return values', () => {
    nodeExecutionStore.completed({
      invocation_source_id: 'call-node',
      result: {
        type: 'workflow_return_output',
        values: { Image: { image_name: 'returned image.png' } },
      },
    });

    expect(nodeExecutionStore.get('call-node')?.outputImageUrl).toBe(
      'https://api.test/api/v1/images/i/returned%20image.png/thumbnail'
    );
  });

  it('settles the named running nodes to the run outcome without disturbing terminal or other nodes', () => {
    nodeExecutionStore.started({ invocation_source_id: 'running' });
    nodeExecutionStore.started({ invocation_source_id: 'other-run' });
    nodeExecutionStore.failed({ error_message: 'failed', invocation_source_id: 'terminal' });

    nodeExecutionStore.settleRunning(['running', 'terminal'], 'completed');

    expect(nodeExecutionStore.get('running')?.status).toBe('completed');
    expect(nodeExecutionStore.get('terminal')?.status).toBe('failed');
    expect(nodeExecutionStore.get('other-run')?.status).toBe('running');

    nodeExecutionStore.settleRunning(['other-run'], 'canceled');

    expect(nodeExecutionStore.get('other-run')).toBeNull();
  });

  it('preserves a failed outcome and error when queue status precedes invocation error', () => {
    nodeExecutionStore.started({ invocation_source_id: 'node-1' });

    nodeExecutionStore.settleRunning(['node-1'], 'failed', 'Workflow failed');

    expect(nodeExecutionStore.get('node-1')).toMatchObject({
      error: 'Workflow failed',
      status: 'failed',
    });
  });
});

describe('node execution origin', () => {
  it('names the run it reflects, notifies origin subscribers, and forgets the origin with the nodes', () => {
    const listener = vi.fn();
    const unsubscribe = nodeExecutionStore.subscribeOrigin(listener);

    nodeExecutionStore.setOrigin({ projectId: 'p', workflowId: 'a' });
    nodeExecutionStore.setOrigin({ projectId: 'p', workflowId: 'a' });
    expect(nodeExecutionStore.getOrigin()).toEqual({ projectId: 'p', workflowId: 'a' });
    expect(listener).toHaveBeenCalledTimes(1);

    nodeExecutionStore.started({ invocation_source_id: 'node-1' });
    nodeExecutionStore.clearAll();
    expect(nodeExecutionStore.getOrigin()).toBeNull();
    expect(nodeExecutionStore.get('node-1')).toBeNull();
    expect(listener).toHaveBeenCalledTimes(2);

    unsubscribe();
  });
});
