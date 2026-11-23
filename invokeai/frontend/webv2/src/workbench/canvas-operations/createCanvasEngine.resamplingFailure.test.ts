import type * as CoreEngineModule from '@workbench/canvas-engine/engine';
import type { CanvasEngineOptions as CoreOptions } from '@workbench/canvas-engine/engine';

import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createCanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => {
  let fail!: (reason: Error) => void;
  const load = new Promise<void>((_resolve, reject) => {
    fail = reject;
  });
  return { fail, load, loads: 0, resample: undefined as CoreOptions['resampleTransform'] };
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
  throw new Error('unreachable');
});

import { createCanvasEngine } from './createCanvasEngine';

it('observes a dynamic import that rejects after the operation was cancelled', async () => {
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
  const controller = new AbortController();
  const pending = mocks.resample!(backend.createSurface(6, 4), { width: 3, height: 2 }, 'lanczos', controller.signal);
  const cancelled = expect(pending).rejects.toMatchObject({ name: 'AbortError' });
  await vi.waitFor(() => expect(mocks.loads).toBe(1));
  controller.abort();
  await cancelled;
  mocks.fail(new Error('late module failure'));
  // Let the failed import settle; Vitest reports any unhandled rejection from it as a failure.
  await new Promise((resolve) => {
    setTimeout(resolve, 0);
  });
  engine.lifecycle.dispose();
});
