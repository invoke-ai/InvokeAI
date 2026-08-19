import type { QueueWorkflowRunCompletedEvent } from '@features/queue/contracts';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { describe, expect, it, vi } from 'vitest';

import type { RunCaptureDeps } from './runCapture';

import { createWorkflowRunCaptureSink } from './runCapture';

const createDeferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });

  return { promise, resolve };
};

const createEvent = (overrides?: Partial<QueueWorkflowRunCompletedEvent>): QueueWorkflowRunCompletedEvent => ({
  imageNames: ['first.png', 'last.png'],
  libraryWorkflowId: 'workflow-1',
  projectId: 'project-1',
  queueItemId: 'queue-item-1',
  ...overrides,
});

const userWorkflow = (workflowId: string) => ({ id: workflowId, meta: { category: 'user', version: '3.0.0' } });

/** The sink is fire-and-forget, so tests wait for its detached chain to settle. */
const settle = async (): Promise<void> => {
  for (let tick = 0; tick < 5; tick += 1) {
    await new Promise<void>((resolve) => {
      setTimeout(resolve, 0);
    });
  }
};

/** `calls` is a parameter so a test can seed it into its own dep overrides. */
const createHarness = (overrides?: Partial<RunCaptureDeps>, calls: string[] = []) => {
  const blob = new Blob(['thumbnail'], { type: 'image/png' });
  const deps: RunCaptureDeps = {
    fetchThumbnailBlob: vi.fn((imageName: string) => {
      calls.push(`fetch:${imageName}`);
      return Promise.resolve(blob);
    }),
    getWorkflow: vi.fn((workflowId: string) => {
      calls.push(`getWorkflow:${workflowId}`);
      return Promise.resolve(userWorkflow(workflowId));
    }),
    invalidateCache: vi.fn(() => {
      calls.push('invalidate');
    }),
    setThumbnail: vi.fn((workflowId: string) => {
      calls.push(`setThumbnail:${workflowId}`);
      return Promise.resolve();
    }),
    touchLastRunAt: vi.fn((workflowId: string) => {
      calls.push(`touchLastRunAt:${workflowId}`);
      return Promise.resolve();
    }),
    ...overrides,
  };

  return { blob, calls, deps, sink: createWorkflowRunCaptureSink(deps) };
};

describe('workflow run capture sink', () => {
  it('uploads the last result thumbnail, stamps last_run_at, then invalidates the library', async () => {
    const { blob, calls, deps, sink } = createHarness();

    sink.onWorkflowRunCompleted(createEvent());
    await settle();

    expect(calls).toEqual([
      'getWorkflow:workflow-1',
      // The run's LAST image is the workflow's cover, not an early intermediate.
      'fetch:last.png',
      'setThumbnail:workflow-1',
      'touchLastRunAt:workflow-1',
      'invalidate',
    ]);
    expect(deps.setThumbnail).toHaveBeenCalledWith('workflow-1', blob, expect.any(AbortSignal));
  });

  it('never touches a default-category workflow, which the account does not own', async () => {
    const { calls, deps, sink } = createHarness({
      getWorkflow: vi.fn().mockResolvedValue({ id: 'workflow-1', meta: { category: 'default', version: '3.0.0' } }),
    });

    sink.onWorkflowRunCompleted(createEvent());
    await settle();

    expect(deps.getWorkflow).toHaveBeenCalledWith('workflow-1', expect.any(AbortSignal));
    // Nothing after the category check ran: no download, no upload, no stamp.
    expect(calls).toEqual([]);
  });

  it('serializes captures per workflow and keeps only the latest queued run', async () => {
    const gate = createDeferred<Blob>();
    const blob = new Blob(['thumbnail'], { type: 'image/png' });
    const calls: string[] = [];
    let fetchCount = 0;
    const { sink } = createHarness(
      {
        fetchThumbnailBlob: vi.fn((imageName: string) => {
          calls.push(`fetch:${imageName}`);
          fetchCount += 1;
          // Hold the first capture open so the next two events queue behind it.
          return fetchCount === 1 ? gate.promise : Promise.resolve(blob);
        }),
      },
      calls
    );

    sink.onWorkflowRunCompleted(createEvent({ imageNames: ['run-1.png'] }));
    await settle();
    sink.onWorkflowRunCompleted(createEvent({ imageNames: ['run-2.png'] }));
    sink.onWorkflowRunCompleted(createEvent({ imageNames: ['run-3.png'] }));
    await settle();

    // Nothing from the queued runs ran while the first capture was in flight.
    expect(calls).toEqual(['getWorkflow:workflow-1', 'fetch:run-1.png']);

    gate.resolve(blob);
    await settle();

    // run-2 was superseded by run-3 before it ever started.
    expect(calls).toEqual([
      'getWorkflow:workflow-1',
      'fetch:run-1.png',
      'setThumbnail:workflow-1',
      'touchLastRunAt:workflow-1',
      'invalidate',
      'getWorkflow:workflow-1',
      'fetch:run-3.png',
      'setThumbnail:workflow-1',
      'touchLastRunAt:workflow-1',
      'invalidate',
    ]);
  });

  it('stops silently when the thumbnail fetch fails, leaving last_run_at untouched', async () => {
    const { calls, deps, sink } = createHarness({
      fetchThumbnailBlob: vi.fn().mockRejectedValue(new Error('404 thumbnail')),
    });

    sink.onWorkflowRunCompleted(createEvent());
    await settle();

    expect(calls).toEqual(['getWorkflow:workflow-1']);
    expect(deps.setThumbnail).not.toHaveBeenCalled();
    expect(deps.touchLastRunAt).not.toHaveBeenCalled();
    expect(deps.invalidateCache).not.toHaveBeenCalled();
  });

  it('stops silently when the thumbnail upload fails, leaving last_run_at untouched', async () => {
    const { deps, sink } = createHarness({
      setThumbnail: vi.fn().mockRejectedValue(new Error('413 payload too large')),
    });

    sink.onWorkflowRunCompleted(createEvent());
    await settle();

    expect(deps.touchLastRunAt).not.toHaveBeenCalled();
    expect(deps.invalidateCache).not.toHaveBeenCalled();
  });

  it('keeps draining queued runs after one of them fails', async () => {
    const { deps, sink } = createHarness({
      getWorkflow: vi.fn().mockRejectedValueOnce(new Error('offline')).mockResolvedValue(userWorkflow('workflow-1')),
    });

    sink.onWorkflowRunCompleted(createEvent({ imageNames: ['run-1.png'] }));
    sink.onWorkflowRunCompleted(createEvent({ imageNames: ['run-2.png'] }));
    await settle();

    expect(deps.setThumbnail).toHaveBeenCalledTimes(1);
    expect(deps.fetchThumbnailBlob).toHaveBeenCalledWith('run-2.png', expect.any(AbortSignal));
  });

  it('ignores an event that carries no result images', async () => {
    const { calls, sink } = createHarness();

    sink.onWorkflowRunCompleted(createEvent({ imageNames: [] }));
    await settle();

    expect(calls).toEqual([]);
  });

  it('captures concurrently when the runs belong to different library records', async () => {
    const gate = createDeferred<Blob>();
    const blob = new Blob(['thumbnail'], { type: 'image/png' });
    const { deps, sink } = createHarness({
      fetchThumbnailBlob: vi.fn((imageName: string) => (imageName === 'a.png' ? gate.promise : Promise.resolve(blob))),
    });

    sink.onWorkflowRunCompleted(createEvent({ imageNames: ['a.png'], libraryWorkflowId: 'workflow-a' }));
    sink.onWorkflowRunCompleted(createEvent({ imageNames: ['b.png'], libraryWorkflowId: 'workflow-b' }));
    await settle();

    // workflow-b never waits behind workflow-a's stalled fetch.
    expect(deps.setThumbnail).toHaveBeenCalledWith('workflow-b', blob, expect.any(AbortSignal));
    gate.resolve(blob);
    await settle();
  });

  it('drops a capture whose account scope expired mid-flight', async () => {
    const gate = createDeferred<Record<string, unknown>>();
    const { deps, sink } = createHarness({ getWorkflow: vi.fn(() => gate.promise) });

    sink.onWorkflowRunCompleted(createEvent());
    await settle();
    accountLifecycle.invalidate();
    gate.resolve(userWorkflow('workflow-1'));
    await settle();

    expect(deps.setThumbnail).not.toHaveBeenCalled();
    expect(deps.touchLastRunAt).not.toHaveBeenCalled();
  });
});
