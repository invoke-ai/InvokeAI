import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { EnqueueGenerateRequest, EnqueueWorkflowRequest } from './types';

const mocks = vi.hoisted(() => ({
  apiFetchJson: vi.fn(),
}));

vi.mock('../backend/http', () => ({
  absolutizeApiUrl: (url: string) => url,
  ApiError: class ApiError extends Error {
    status: number;

    constructor(status: number) {
      super('API error');
      this.status = status;
    }
  },
  apiFetchJson: mocks.apiFetchJson,
}));

const createRequest = (overrides: Partial<EnqueueGenerateRequest> = {}): EnqueueGenerateRequest => ({
  batchCount: 3,
  destination: 'gallery',
  graph: { edges: [], id: 'graph-1', nodes: {} },
  negativePrompt: 'low quality',
  negativePromptNodeId: 'negative_prompt',
  positivePrompt: 'a fjord at dawn',
  positivePromptNodeId: 'positive_prompt',
  seed: 10,
  seedNodeId: 'seed',
  shouldRandomizeSeed: false,
  sourceQueueItemId: 'local-1',
  ...overrides,
});

const createWorkflowRequest = (overrides: Partial<EnqueueWorkflowRequest> = {}): EnqueueWorkflowRequest => ({
  batchCount: 2,
  destination: 'gallery',
  graph: { edges: [], id: 'graph-1', nodes: {} },
  sourceQueueItemId: 'local-1',
  ...overrides,
});

const getSubmittedBody = () => {
  const init = mocks.apiFetchJson.mock.calls[0]?.[1] as RequestInit | undefined;

  expect(init?.body).toEqual(expect.any(String));

  return JSON.parse(init?.body as string) as {
    batch: {
      data: { field_name: string; items: unknown[]; node_path: string }[][];
      runs: number;
    };
  };
};

describe('enqueueGenerateGraph', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
    mocks.apiFetchJson.mockResolvedValue({ batch: { batch_id: 'batch-1' }, item_ids: [1, 2, 3] });
  });

  it('keeps a fixed seed for every run when randomization is disabled', async () => {
    const { enqueueGenerateGraph } = await import('./api');

    await enqueueGenerateGraph(createRequest({ shouldRandomizeSeed: false }));

    const body = getSubmittedBody();

    expect(body.batch.runs).toBe(3);
    expect(body.batch.data[0]).toEqual([
      { field_name: 'value', items: [10], node_path: 'seed' },
      { field_name: 'value', items: ['a fjord at dawn'], node_path: 'positive_prompt' },
      { field_name: 'value', items: ['low quality'], node_path: 'negative_prompt' },
    ]);
  });

  it('expands seeds per batch item only when randomization is enabled', async () => {
    const { enqueueGenerateGraph } = await import('./api');

    await enqueueGenerateGraph(createRequest({ shouldRandomizeSeed: true }));

    const body = getSubmittedBody();

    expect(body.batch.runs).toBe(1);
    expect(body.batch.data[0]).toEqual([
      { field_name: 'value', items: [10, 11, 12], node_path: 'seed' },
      {
        field_name: 'value',
        items: ['a fjord at dawn', 'a fjord at dawn', 'a fjord at dawn'],
        node_path: 'positive_prompt',
      },
      { field_name: 'value', items: ['low quality', 'low quality', 'low quality'], node_path: 'negative_prompt' },
    ]);
  });
});

describe('enqueueWorkflowGraph', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
    mocks.apiFetchJson.mockResolvedValue({ batch: { batch_id: 'batch-1' }, item_ids: [1, 2] });
  });

  it('submits workflow runs for the requested batch count', async () => {
    const { enqueueWorkflowGraph } = await import('./api');

    await enqueueWorkflowGraph(createWorkflowRequest());

    expect(getSubmittedBody().batch.runs).toBe(2);
  });

  it('does not cap workflow runs', async () => {
    const { enqueueWorkflowGraph } = await import('./api');

    await enqueueWorkflowGraph(createWorkflowRequest({ batchCount: 10_000 }));

    expect(getSubmittedBody().batch.runs).toBe(10_000);
  });
});
