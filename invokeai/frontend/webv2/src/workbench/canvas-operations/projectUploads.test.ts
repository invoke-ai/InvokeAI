import { accountLifecycle } from '@platform/state/accountLifecycle';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createCanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { canvasApplicationPort } from './applicationPort';
import { CanvasImageUploadError } from './backend/canvasImages';
import { createCanvasEngine } from './createCanvasEngine';
import { getCanvasOperations } from './operationAccess';

const createEngine = (ensureProjectOnServer: () => Promise<void>) => {
  const store = createWorkbenchStore();
  const projectId = store.getState().activeProjectId;
  return createCanvasEngine({
    backend: createTestStubRasterBackend(),
    ensureProjectOnServer,
    imageResolver: () => Promise.resolve(new Blob()),
    mutationPort: createCanvasProjectMutationPort(store, projectId),
    projectId,
    reportError: () => undefined,
  });
};

beforeEach(() => {
  accountLifecycle.activate('canvas-upload-test');
});
afterEach(() => vi.restoreAllMocks());

describe('project-owned canvas uploads', () => {
  it('waits for project acknowledgement before uploading utility inputs and generation composites', async () => {
    let acknowledge!: () => void;
    const ready = new Promise<void>((resolve) => {
      acknowledge = resolve;
    });
    const engine = createEngine(() => ready);
    const upload = vi.spyOn(canvasApplicationPort, 'uploadImage').mockResolvedValue({
      height: 8,
      imageName: 'intermediate.png',
      width: 8,
    });
    try {
      const utility = getCanvasOperations(engine).uploadIntermediate(new Blob());
      const composite = engine.exports.getCompositeExecutorDeps().uploadImage(new Blob());
      await Promise.resolve();
      expect(upload).not.toHaveBeenCalled();
      acknowledge();
      await Promise.all([utility, composite]);
      expect(upload).toHaveBeenCalledTimes(2);
      for (const [, options] of upload.mock.calls) {
        expect(options).toMatchObject({ isIntermediate: true, projectId: engine.projectId });
      }
    } finally {
      engine.lifecycle.dispose();
    }
  });

  it('uploads without provenance when the server has not accepted the project, and with it once it has', async () => {
    const ensure = vi
      .fn()
      .mockRejectedValueOnce(Object.assign(new Error('creation failed'), { reason: 'unsynced' }))
      .mockResolvedValue(undefined);
    const engine = createEngine(ensure);
    const upload = vi.spyOn(canvasApplicationPort, 'uploadImage').mockResolvedValue({
      height: 8,
      imageName: 'intermediate.png',
      width: 8,
    });
    try {
      await getCanvasOperations(engine).uploadIntermediate(new Blob());
      expect(upload.mock.calls[0]![1]).toMatchObject({ isIntermediate: true, projectId: undefined });
      await getCanvasOperations(engine).uploadIntermediate(new Blob());
      expect(upload.mock.calls[1]![1]).toMatchObject({ isIntermediate: true, projectId: engine.projectId });
    } finally {
      engine.lifecycle.dispose();
    }
  });

  it('retries without provenance only when the server no longer knows the project', async () => {
    const engine = createEngine(() => Promise.resolve());
    const upload = vi
      .spyOn(canvasApplicationPort, 'uploadImage')
      .mockRejectedValueOnce(new CanvasImageUploadError('{"detail":"Project not found"}', 404))
      .mockResolvedValueOnce({ height: 8, imageName: 'unowned.png', width: 8 })
      .mockRejectedValueOnce(new CanvasImageUploadError('{"detail":"Board not found"}', 404));
    try {
      await expect(getCanvasOperations(engine).uploadIntermediate(new Blob())).resolves.toMatchObject({
        imageName: 'unowned.png',
      });
      expect(upload.mock.calls.map(([, options]) => options?.projectId)).toEqual([engine.projectId, undefined]);

      await expect(getCanvasOperations(engine).uploadIntermediate(new Blob())).rejects.toMatchObject({ status: 404 });
      expect(upload).toHaveBeenCalledTimes(3);
    } finally {
      engine.lifecycle.dispose();
    }
  });

  it('sends later uploads once without provenance after a missing project, and tries it again later', async () => {
    let now = 1_000_000;
    vi.spyOn(Date, 'now').mockImplementation(() => now);
    const engine = createEngine(() => Promise.resolve());
    const upload = vi
      .spyOn(canvasApplicationPort, 'uploadImage')
      .mockRejectedValueOnce(new CanvasImageUploadError('{"detail":"Project not found"}', 404))
      .mockResolvedValue({ height: 8, imageName: 'unowned.png', width: 8 });
    try {
      await getCanvasOperations(engine).uploadIntermediate(new Blob());
      await getCanvasOperations(engine).uploadIntermediate(new Blob());
      expect(upload.mock.calls.map(([, options]) => options?.projectId)).toEqual([
        engine.projectId,
        undefined,
        undefined,
      ]);

      now += 60_000;
      await getCanvasOperations(engine).uploadIntermediate(new Blob());
      expect(upload.mock.calls.at(-1)![1]).toMatchObject({ projectId: engine.projectId });
    } finally {
      engine.lifecycle.dispose();
    }
  });

  it('does not upload for a project that closed while it waited', async () => {
    const engine = createEngine(() =>
      Promise.reject(new DOMException('The canvas project is no longer open.', 'AbortError'))
    );
    const upload = vi.spyOn(canvasApplicationPort, 'uploadImage');
    try {
      await expect(getCanvasOperations(engine).uploadIntermediate(new Blob())).rejects.toMatchObject({
        name: 'AbortError',
      });
      expect(upload).not.toHaveBeenCalled();
    } finally {
      engine.lifecycle.dispose();
    }
  });

  it.each(['account change', 'engine disposal', 'operation cancellation'] as const)(
    'stops waiting for the project and never uploads after %s',
    async (cancellation) => {
      let acknowledge!: () => void;
      const ready = new Promise<void>((resolve) => {
        acknowledge = resolve;
      });
      const engine = createEngine(() => ready);
      const controller = new AbortController();
      const upload = vi.spyOn(canvasApplicationPort, 'uploadImage');
      try {
        const pending = getCanvasOperations(engine).uploadIntermediate(new Blob(), controller.signal);
        const rejected = expect(pending).rejects.toMatchObject({
          name: cancellation === 'account change' ? 'AccountScopeExpiredError' : 'AbortError',
        });
        if (cancellation === 'account change') {
          accountLifecycle.activate('another-account');
        } else if (cancellation === 'engine disposal') {
          engine.lifecycle.dispose();
        } else {
          controller.abort();
        }
        await rejected;
        acknowledge();
        await Promise.resolve();
        expect(upload).not.toHaveBeenCalled();
      } finally {
        engine.lifecycle.dispose();
      }
    }
  );
});
