import type * as CoreEngineModule from '@workbench/canvas-engine/engine';
import type { CanvasEngineOptions as CoreOptions } from '@workbench/canvas-engine/engine';

import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createCanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => {
  let release!: () => void;
  const load = new Promise<void>((resolve) => {
    release = resolve;
  });
  return {
    load,
    release,
    loads: 0,
    resample: undefined as CoreOptions['resampleTransform'],
    run: vi.fn<NonNullable<CoreOptions['resampleTransform']>>(() =>
      Promise.resolve({ width: 3, height: 2 } as ImageData)
    ),
    create: vi.fn(),
  };
});

vi.mock('@workbench/canvas-engine/engine', async (importOriginal) => {
  const actual = await importOriginal<typeof CoreEngineModule>();
  return {
    ...actual,
    createCanvasEngine: (options: CoreOptions) => {
      mocks.resample = options.resampleTransform;
      return actual.createCanvasEngine(options);
    },
  };
});

vi.mock('./transformResampling', async () => {
  mocks.loads++;
  await mocks.load;
  return {
    createTransformResampler: mocks.create.mockImplementation(() => mocks.run),
  };
});

import { createCanvasEngine } from './createCanvasEngine';

describe('canvas transform resampling loading', () => {
  it('loads on first use and refuses an operation cancelled while loading', async () => {
    const store = createWorkbenchStore();
    const projectId = store.getState().activeProjectId;
    const backend = createTestStubRasterBackend();
    const engine = createCanvasEngine({
      backend,
      ensureProjectOnServer: () => Promise.resolve(),
      imageResolver: () => Promise.resolve(new Blob()),
      mutationPort: createCanvasProjectMutationPort(store, projectId),
      projectId,
      reportError: () => undefined,
    });
    const surface = backend.createSurface(6, 4);
    const target = { width: 3, height: 2 };
    expect(mocks.loads).toBe(0);
    const cancelled = new AbortController();
    const pending = mocks.resample!(surface, target, 'lanczos', cancelled.signal);
    const cancelledResult = expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    await vi.waitFor(() => expect(mocks.loads).toBe(1));
    cancelled.abort();
    // The dynamic import is still pending here: cancellation must not wait for its resolution.
    await cancelledResult;
    expect(mocks.create).not.toHaveBeenCalled();
    expect(mocks.run).not.toHaveBeenCalled();
    const resample = mocks.resample!;
    const secondEngine = createCanvasEngine({
      backend,
      ensureProjectOnServer: () => Promise.resolve(),
      imageResolver: () => Promise.resolve(new Blob()),
      mutationPort: createCanvasProjectMutationPort(store, projectId),
      projectId,
      reportError: () => undefined,
    });
    const pendingDispose = mocks.resample!(surface, target, 'lanczos', new AbortController().signal);
    const disposedResult = expect(pendingDispose).rejects.toMatchObject({ name: 'AbortError' });
    secondEngine.lifecycle.dispose();
    await disposedResult;
    expect(mocks.create).not.toHaveBeenCalled();
    mocks.release();

    const signal = new AbortController().signal;
    await expect(resample(surface, target, 'bicubic', signal)).resolves.toMatchObject(target);
    expect(mocks.loads).toBe(1);
    expect(mocks.run).toHaveBeenCalledExactlyOnceWith(surface, target, 'bicubic', signal, undefined);
    const onProgress = vi.fn();
    mocks.run.mockImplementationOnce((_surface, _target, _mode, _signal, progress) => {
      progress?.('queued');
      progress?.('resampling');
      return Promise.resolve({ width: 3, height: 2 } as ImageData);
    });
    await expect(resample(surface, target, 'lanczos', signal, onProgress)).resolves.toMatchObject(target);
    expect(mocks.run).toHaveBeenCalledTimes(2);
    expect(mocks.run).toHaveBeenNthCalledWith(2, surface, target, 'lanczos', signal, onProgress);
    expect(onProgress.mock.calls).toEqual([['queued'], ['resampling']]);
    expect(mocks.loads).toBe(1);
    engine.lifecycle.dispose();
  });
});
