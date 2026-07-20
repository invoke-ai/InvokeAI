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
      progress: null,
      progressMessage: null,
      status: 'failed',
    });
  });

  it('settles every running node without disturbing terminal state', () => {
    nodeExecutionStore.started({ invocation_source_id: 'running' });
    nodeExecutionStore.failed({ error_message: 'failed', invocation_source_id: 'terminal' });

    nodeExecutionStore.settleRunning();

    expect(nodeExecutionStore.get('running')?.status).toBe('completed');
    expect(nodeExecutionStore.get('terminal')?.status).toBe('failed');
  });
});
