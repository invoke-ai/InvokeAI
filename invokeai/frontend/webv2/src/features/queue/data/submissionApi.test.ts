import type { QueueEnqueueGenerateRequest, QueueEnqueueWorkflowRequest } from '@features/queue/core/types';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  apiFetch: vi.fn(),
  apiFetchJson: vi.fn(),
}));

vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  absolutizeApiUrl: (url: string) => url,
  ApiError: class ApiError extends Error {
    status: number;

    constructor(message: string, status: number) {
      super(message);
      this.status = status;
    }
  },
  apiFetch: mocks.apiFetch,
  apiFetchJson: mocks.apiFetchJson,
}));

describe('acknowledgeQueueEnqueue', () => {
  it('releases the exact enqueue receipt', async () => {
    const { acknowledgeQueueEnqueue } = await import('./submissionApi');

    await acknowledgeQueueEnqueue('project-1', 'local-1');

    expect(mocks.apiFetch).toHaveBeenCalledWith('/api/v1/queue/default/enqueue_batch/acknowledge', {
      body: JSON.stringify({ idempotency_key: 'webv2:project-1:local-1' }),
      headers: { 'Content-Type': 'application/json' },
      method: 'POST',
    });
  });
});

describe('getQueueEnqueueReceipt', () => {
  it.each([
    { batch_id: 'batch-1', enqueued: 1, item_ids: [], requested: 1 },
    { batch_id: 'batch-1', enqueued: 2, item_ids: [11, 11], requested: 2 },
    { batch_id: '', enqueued: 1, item_ids: [11], requested: 1 },
    { batch_id: 'batch-1', enqueued: 1, item_ids: [-1], requested: 1 },
    { batch_id: 'batch-1', enqueued: 2, item_ids: [11, 12], requested: 1 },
    { batch_id: 'batch-1', enqueued: 0, item_ids: [], requested: 1 },
  ])('rejects malformed accepted receipts: %j', async (receipt) => {
    const { getQueueEnqueueReceipt } = await import('./submissionApi');
    mocks.apiFetchJson.mockResolvedValueOnce(receipt);
    await expect(getQueueEnqueueReceipt('project-1', 'local-1')).rejects.toThrow('Invalid queue enqueue response');
  });

  it('loads only the caller-owned receipt and maps its accepted ids', async () => {
    const { getQueueEnqueueReceipt } = await import('./submissionApi');
    mocks.apiFetchJson.mockResolvedValueOnce({ batch_id: 'batch-1', enqueued: 2, item_ids: [11, 12], requested: 3 });

    await expect(getQueueEnqueueReceipt('project-1', 'local-1')).resolves.toEqual({
      batchId: 'batch-1',
      enqueued: 2,
      itemIds: [11, 12],
      requested: 3,
    });
    expect(mocks.apiFetchJson).toHaveBeenLastCalledWith(
      '/api/v1/queue/default/enqueue_batch/receipt?idempotency_key=webv2%3Aproject-1%3Alocal-1'
    );
  });

  it('treats only 404 as an absent receipt', async () => {
    const { getQueueEnqueueReceipt } = await import('./submissionApi');
    const { ApiError } = await import('@platform/transport/http');
    mocks.apiFetchJson.mockRejectedValueOnce(new ApiError('not found', 404));
    await expect(getQueueEnqueueReceipt('project-1', 'local-1')).resolves.toBeNull();
    const offline = new Error('offline');
    mocks.apiFetchJson.mockRejectedValueOnce(offline);
    await expect(getQueueEnqueueReceipt('project-1', 'local-1')).rejects.toBe(offline);
  });
});

const createRequest = (overrides: Partial<QueueEnqueueGenerateRequest> = {}): QueueEnqueueGenerateRequest => ({
  batchCount: 3,
  destination: 'gallery',
  graph: { edges: [], id: 'graph-1', nodes: {} },
  negativePrompt: 'low quality',
  negativePromptNodeId: 'negative_prompt',
  positivePrompt: 'a fjord at dawn',
  positivePromptNodeId: 'positive_prompt',
  projectId: 'project-1',
  seed: 10,
  seedNodeId: 'seed',
  seedStep: 0,
  sourceQueueItemId: 'local-1',
  ...overrides,
});

const createWorkflowRequest = (overrides: Partial<QueueEnqueueWorkflowRequest> = {}): QueueEnqueueWorkflowRequest => ({
  batchCount: 2,
  destination: 'gallery',
  graph: { edges: [], id: 'graph-1', nodes: {} },
  projectId: 'project-1',
  sourceQueueItemId: 'local-1',
  ...overrides,
});

const getSubmittedBody = () => {
  const init = mocks.apiFetchJson.mock.calls[0]?.[1] as RequestInit | undefined;

  expect(init?.body).toEqual(expect.any(String));

  return JSON.parse(init?.body as string) as {
    prepend: boolean;
    batch: {
      batch_id?: string;
      data: { field_name: string; items: unknown[]; node_path: string }[][];
      idempotency_key: string;
      origin: string;
      runs: number;
      workflow?: Record<string, unknown>;
    };
  };
};

describe('enqueueGenerate', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
    mocks.apiFetchJson.mockResolvedValue({
      batch: { batch_id: 'batch-1' },
      enqueued: 3,
      item_ids: [1, 2, 3],
      requested: 3,
    });
  });

  it('expands seeds per batch item when the seed steps', async () => {
    const { enqueueGenerate } = await import('./submissionApi');

    await enqueueGenerate(createRequest({ seedStep: 1 }));

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

  it('includes project-aware webv2 origin metadata', async () => {
    const { enqueueGenerate } = await import('./submissionApi');

    await enqueueGenerate(createRequest());

    expect(getSubmittedBody().batch.origin).toBe('webv2:p:project-1:q:local-1');
    expect(getSubmittedBody().batch.idempotency_key).toBe('webv2:project-1:local-1');
    expect(getSubmittedBody().batch.batch_id).toBeUndefined();
  });

  describe('expanded prompts', () => {
    it('submits several prompts as their own batch dimension', async () => {
      const { enqueueGenerate } = await import('./submissionApi');

      await enqueueGenerate(
        createRequest({
          batchCount: 2,
          positivePrompt: 'a {red|green} cat',
          positivePrompts: ['a red cat', 'a green cat'],
          seedBehaviour: 'per-iteration',
          seedStep: 1,
        })
      );

      const body = getSubmittedBody();

      expect(body.batch.data).toEqual([
        [{ field_name: 'value', items: [10, 11], node_path: 'seed' }],
        [
          { field_name: 'value', items: ['a red cat', 'a green cat'], node_path: 'positive_prompt' },
          { field_name: 'value', items: ['low quality', 'low quality'], node_path: 'negative_prompt' },
        ],
      ]);
      expect(body.batch.runs).toBe(1);
    });
  });
});

describe('enqueueGenerate legacy replays', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
    mocks.apiFetchJson.mockResolvedValue({
      batch: { batch_id: 'batch-1' },
      enqueued: 1,
      item_ids: [1],
      requested: 1,
    });
  });

  it('replays a pinned two-prompt per-image batch with the seeds that version stepped', async () => {
    const { enqueueGenerate } = await import('./submissionApi');

    await enqueueGenerate(
      createRequest({
        batchCount: 2,
        legacySeedPlan: true,
        positivePrompts: ['a', 'b'],
        seed: 42,
        seedBehaviour: 'per-image',
        seedStep: 0,
      })
    );

    // Before seed modes, sharing disabled always stepped, toggle or not: one zipped group with a seed per image and the
    // prompt list repeated per iteration.
    expect(getSubmittedBody().batch.data).toEqual([
      [
        { field_name: 'value', items: [42, 43, 44, 45], node_path: 'seed' },
        { field_name: 'value', items: ['a', 'b', 'a', 'b'], node_path: 'positive_prompt' },
        {
          field_name: 'value',
          items: ['low quality', 'low quality', 'low quality', 'low quality'],
          node_path: 'negative_prompt',
        },
      ],
    ]);
    expect(getSubmittedBody().batch.runs).toBe(1);
  });
});

describe('enqueueWorkflow', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
    mocks.apiFetchJson.mockResolvedValue({
      batch: { batch_id: 'batch-1' },
      enqueued: 2,
      item_ids: [1, 2],
      requested: 2,
    });
  });

  it('rejects ambiguous accepted ids but accepts an explicit zero-capacity result', async () => {
    const { enqueueWorkflow } = await import('./submissionApi');
    mocks.apiFetchJson.mockResolvedValueOnce({
      batch: { batch_id: 'batch-1' },
      enqueued: 1,
      item_ids: [],
      requested: 2,
    });
    await expect(enqueueWorkflow(createWorkflowRequest())).rejects.toThrow('Invalid queue enqueue response');
    mocks.apiFetchJson.mockResolvedValueOnce({
      batch: { batch_id: 'batch-1' },
      enqueued: 0,
      item_ids: [],
      requested: 2,
    });
    await expect(enqueueWorkflow(createWorkflowRequest())).resolves.toEqual({
      batchId: 'batch-1',
      enqueued: 0,
      itemIds: [],
      requested: 2,
    });
  });

  it('sends batch-node groups in the backend shape and lets it multiply them across runs', async () => {
    const { enqueueWorkflow } = await import('./submissionApi');

    await enqueueWorkflow({
      ...createWorkflowRequest(),
      batchData: [
        [{ fieldName: 'cfg', items: [1, 2], nodeId: 'denoise' }],
        [{ fieldName: 'image', items: [{ image_name: 'a.png' }], nodeId: 'sink' }],
      ],
    });

    expect(getSubmittedBody().batch.runs).toBe(2);
    expect(getSubmittedBody().batch.data).toEqual([
      [{ field_name: 'cfg', items: [1, 2], node_path: 'denoise' }],
      [{ field_name: 'image', items: [{ image_name: 'a.png' }], node_path: 'sink' }],
    ]);
  });

  it('expands recorded seeds into one zipped group and never redraws them', async () => {
    const { enqueueWorkflow } = await import('./submissionApi');
    const seeds = [
      { fieldName: 'seed', nodeId: 'noise-a', seed: 1, seedStep: 1 as const },
      { fieldName: 'seed', nodeId: 'noise-b', seed: 4_294_967_295, seedStep: -1 as const },
    ];

    await enqueueWorkflow(createWorkflowRequest({ batchCount: 3, seeds }));

    expect(getSubmittedBody().batch).toMatchObject({
      data: [
        [
          { field_name: 'seed', items: [1, 2, 3], node_path: 'noise-a' },
          { field_name: 'seed', items: [4_294_967_295, 4_294_967_294, 4_294_967_293], node_path: 'noise-b' },
        ],
      ],
      runs: 1,
    });

    // A single run already carries its seed in the graph; a seedless batch repeats the graph.
    for (const request of [createWorkflowRequest({ batchCount: 1, seeds }), createWorkflowRequest({ batchCount: 2 })]) {
      mocks.apiFetchJson.mockClear();
      await enqueueWorkflow(request);

      expect(getSubmittedBody().batch).not.toHaveProperty('data');
      expect(getSubmittedBody().batch.runs).toBe(request.batchCount);
    }
  });
});

describe('enqueueUtility', () => {
  afterEach(() => vi.useRealTimers());
  const request = {
    graph: { edges: [], id: 'utility', nodes: {} },
    idempotencyKey: 'utility:operation',
    origin: 'webv2:util:operation',
    prepend: true,
  };

  it('recovers accepted admission from a receipt after the response is lost', async () => {
    const { enqueueUtility } = await import('./submissionApi');
    mocks.apiFetchJson.mockReset();
    mocks.apiFetchJson.mockRejectedValueOnce(new TypeError('lost response')).mockResolvedValueOnce({
      batch_id: 'accepted',
      enqueued: 1,
      item_ids: [41],
      requested: 1,
    });
    const result = await enqueueUtility(request);
    expect(result.itemIds).toEqual([41]);
    expect(mocks.apiFetchJson.mock.calls.map(([url]) => url)).toEqual([
      '/api/v1/queue/default/enqueue_batch',
      '/api/v1/queue/default/enqueue_batch/receipt?idempotency_key=utility%3Aoperation',
    ]);
    expect(mocks.apiFetch).not.toHaveBeenCalled();
  });

  it.each([409, 429])('does not replay definitive admission rejection (%s)', async (status) => {
    const { enqueueUtility } = await import('./submissionApi');
    const { ApiError } = await import('@platform/transport/http');
    mocks.apiFetchJson.mockRejectedValue(new ApiError('conflict or receipt quota', status));
    await expect(enqueueUtility(request)).rejects.toMatchObject({ status });
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);
  });

  it('fences admission recovery when the account changes during a lost response', async () => {
    const { enqueueUtility } = await import('./submissionApi');
    mocks.apiFetchJson.mockImplementationOnce(() => {
      accountLifecycle.activate('replacement-account');
      return Promise.reject(new TypeError('lost response'));
    });
    await expect(enqueueUtility(request)).rejects.toThrow('account scope');
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);
  });

  it('acknowledges only on retention and bounds failed acknowledgement attempts', async () => {
    vi.useFakeTimers();
    const { enqueueUtility } = await import('./submissionApi');
    mocks.apiFetch.mockReset().mockRejectedValue(new TypeError('offline'));
    const result = await enqueueUtility(request);
    expect(mocks.apiFetch).not.toHaveBeenCalled();
    result.acknowledge();
    result.acknowledge();
    await vi.advanceTimersByTimeAsync(15_000);
    expect(mocks.apiFetch).toHaveBeenCalledTimes(5);
    expect(vi.getTimerCount()).toBe(0);
  });

  beforeEach(() => {
    accountLifecycle.activate('utility-account');
    mocks.apiFetch.mockReset().mockResolvedValue(undefined);
    mocks.apiFetchJson.mockReset();
    mocks.apiFetchJson.mockResolvedValue({ enqueued: 1, item_ids: [4] });
  });
});

describe('getResultImages', () => {
  beforeEach(() => {
    accountLifecycle.activate('test-account');
    mocks.apiFetchJson.mockReset();
  });

  it.each(['image', 'video'] as const)('bounds %s metadata requests for collection-sized results', async (kind) => {
    let active = 0;
    let peak = 0;
    mocks.apiFetchJson.mockImplementation(async (url: string) => {
      if (url === '/api/v1/queue/default/i/1') {
        return {
          item_id: 1,
          status: 'completed',
          session: {
            results: Object.fromEntries(
              Array.from({ length: 20 }, (_, index) => [
                String(index),
                {
                  [kind]: { [`${kind}_name`]: `asset-${index}` },
                },
              ])
            ),
          },
        };
      }
      active += 1;
      peak = Math.max(peak, active);
      await new Promise<void>((resolve) => {
        setTimeout(resolve, 1);
      });
      active -= 1;
      return {
        image_name: url.split('/').at(-1),
        image_url: url,
        thumbnail_url: url,
        height: 512,
        width: 512,
        is_intermediate: false,
      };
    });
    const { getResultImages, getResultVideoNames } = await import('./submissionApi');
    const result =
      kind === 'image'
        ? await getResultImages(1, 'source-1', '2026-09-04T00:00:00Z')
        : await getResultVideoNames(1, { excludeIntermediate: true });
    expect(result).toHaveLength(20);
    expect(peak).toBeLessThanOrEqual(8);
  });

  it('extracts image names from image collection outputs', async () => {
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url === '/api/v1/queue/default/i/1') {
        return Promise.resolve({
          item_id: 1,
          session: {
            results: {
              external: {
                collection: [{ image_name: 'external-a.png' }, { image_name: 'external-b.png' }],
              },
            },
          },
          status: 'completed',
        });
      }

      const imageName = decodeURIComponent(url.replace('/api/v1/images/i/', ''));
      return Promise.resolve({
        height: 768,
        image_name: imageName,
        image_url: `/images/${imageName}`,
        is_intermediate: false,
        thumbnail_url: `/thumbs/${imageName}`,
        width: 1024,
      });
    });

    const { getResultImages } = await import('./submissionApi');

    const images = await getResultImages(1, 'source-1', '2026-06-15T00:00:00.000Z');

    expect(images.map((image) => image.imageName)).toEqual(['external-a.png', 'external-b.png']);
  });

  it('does not issue image requests from queue data resolved after an account switch', async () => {
    accountLifecycle.activate('user-a');
    let resolveItem: ((item: unknown) => void) | undefined;
    mocks.apiFetchJson.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveItem = resolve;
        })
    );
    const { getResultImages } = await import('./submissionApi');

    const oldResult = getResultImages(1, 'source-a', '2026-06-15T00:00:00.000Z');
    const signal = (mocks.apiFetchJson.mock.calls[0]?.[1] as RequestInit | undefined)?.signal;

    accountLifecycle.invalidate();
    accountLifecycle.activate('user-b');
    resolveItem?.({
      item_id: 1,
      session: { results: { output: { image: { image_name: 'user-a.png' } } } },
      status: 'completed',
    });

    await expect(oldResult).rejects.toThrow('no longer active');
    expect(signal?.aborted).toBe(true);
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);
  });
});

describe('getResultVideoNames', () => {
  beforeEach(() => {
    accountLifecycle.activate('test-account');
    mocks.apiFetchJson.mockReset();
  });

  it('extracts video names from video outputs, ignoring image outputs', async () => {
    mocks.apiFetchJson.mockResolvedValue({
      item_id: 1,
      session: {
        results: {
          decode: { image: { image_name: 'frame.png' }, type: 'image_output' },
          l2v: {
            duration: 5.17,
            type: 'video_output',
            video: { video_name: 'clip-1.mp4' },
          },
          l2v_again: { type: 'video_output', video: { video_name: 'clip-1.mp4' } },
        },
      },
      status: 'completed',
    });

    const { getResultVideoNames } = await import('./submissionApi');

    await expect(getResultVideoNames(1)).resolves.toEqual(['clip-1.mp4']);
    // Names only — no per-video DTO hydration requests.
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);
  });
});

describe('readQueueItemLastSavedOutput', () => {
  const imageUrl = (imageName: string) => `/api/v1/images/i/${imageName}`;
  const serveItem = (
    item: Record<string, unknown>,
    intermediates: readonly string[],
    missing: readonly string[] = []
  ) =>
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url === '/api/v1/queue/default/i/1') {
        return Promise.resolve({ item_id: 1, status: 'completed', ...item });
      }

      const [, kind, name] = /^\/api\/v1\/(image|video)s\/i\/(.+)$/.exec(url)!;

      if (missing.includes(name!)) {
        return Promise.reject(new ApiError('Not found', 404));
      }

      return Promise.resolve({ [`${kind}_name`]: name, is_intermediate: intermediates.includes(name!) });
    });
  const session = {
    results: {
      first: { image: { image_name: 'first.png' }, type: 'image_output' },
      saved: { image: { image_name: 'saved.png' }, type: 'image_output' },
      deleted: { image: { image_name: 'deleted.png' }, type: 'image_output' },
      preview: { image: { image_name: 'preview.png' }, type: 'image_output' },
    },
  };
  let ApiError: new (message: string, status: number) => Error;

  beforeEach(async () => {
    accountLifecycle.activate('test-account');
    mocks.apiFetchJson.mockReset();
    ({ ApiError } = (await import('@platform/transport/http')) as unknown as { ApiError: typeof ApiError });
  });

  it('walks back from the last output to the first saved image, reading no earlier record', async () => {
    serveItem({ session }, ['preview.png'], ['deleted.png']);
    const { readQueueItemLastSavedOutput } = await import('./submissionApi');

    await expect(readQueueItemLastSavedOutput(1)).resolves.toEqual({ kind: 'image', name: 'saved.png' });
    expect(mocks.apiFetchJson.mock.calls.map(([url]) => url)).not.toContain(imageUrl('first.png'));
  });

  it.each([
    ['a video-only item', { clip: { type: 'video_output', video: { video_name: 'clip.mp4' } } }, [], 'clip.mp4'],
    [
      'an item whose frame extract is intermediate',
      {
        frame: { image: { image_name: 'frame.png' }, type: 'image_output' },
        clip: { type: 'video_output', video: { video_name: 'clip.mp4' } },
        last_frame: { image: { image_name: 'last_frame.png' }, type: 'image_output' },
      },
      ['frame.png', 'last_frame.png'],
      'clip.mp4',
    ],
  ])('reveals the saved video of %s', async (_, results, intermediates, videoName) => {
    serveItem({ session: { results } }, intermediates);
    const { readQueueItemLastSavedOutput } = await import('./submissionApi');

    await expect(readQueueItemLastSavedOutput(1)).resolves.toEqual({ kind: 'video', name: videoName });
  });
});
