import type { CanvasRasterLayerContractV2 } from '@workbench/types';

import { describe, expect, it, vi } from 'vitest';

import type { LayerExportGuard } from './engine';

import { createCanvasOperationController } from './canvasOperationController';

const layer: CanvasRasterLayerContractV2 = {
  blendMode: 'normal',
  id: 'layer-1',
  isEnabled: true,
  isLocked: false,
  name: 'Layer',
  opacity: 1,
  source: { fill: '#fff', height: 10, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 10 },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
};

const guard: LayerExportGuard = {
  cacheVersion: 1,
  documentGeneration: 1,
  layer,
  layerId: layer.id,
  projectId: 'project-1',
};

const createDeferred = <T>() => {
  let resolve!: (value: T) => void;
  let reject!: (cause: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, reject, resolve };
};

const filterIdentity = { kind: 'filter' as const, layerId: layer.id, projectId: 'project-1' };
const selectObjectIdentity = { kind: 'select-object' as const, projectId: 'project-1' };

const compositeGuard = {
  bbox: { height: 10, width: 10, x: 0, y: 0 },
  candidates: [{ cacheVersion: 1, layer, layerId: layer.id }],
  documentFingerprint: 'document-1',
  documentGeneration: 1,
  participants: [{ cacheVersion: 1, layer, layerId: layer.id }],
  projectId: 'project-1',
};
const documentSelectObjectIdentity = { kind: 'select-object' as const, projectId: 'project-1' };

describe('createCanvasOperationController', () => {
  it('uses document-composite identity and guards for Select Object without a layer sentinel', () => {
    const controller = createCanvasOperationController({ isGuardCurrent: () => true });

    const session = controller.start({
      cleanupPreview: vi.fn(),
      guard: compositeGuard,
      identity: documentSelectObjectIdentity,
    });

    expect(session).not.toBeNull();
    expect(controller.getSnapshot()).toMatchObject({ identity: documentSelectObjectIdentity, status: 'active' });
    // @ts-expect-error Select Object cannot accept a layer guard.
    expect(controller.start({ cleanupPreview: vi.fn(), guard, identity: documentSelectObjectIdentity })).toBeNull();
    // @ts-expect-error Filter cannot accept a document-composite guard.
    expect(controller.start({ cleanupPreview: vi.fn(), guard: compositeGuard, identity: filterIdentity })).toBeNull();
  });
  it('replaces the active operation and prevents its stale session from affecting the replacement', async () => {
    const firstWork = createDeferred<string>();
    const firstCleanup = vi.fn();
    const secondCleanup = vi.fn();
    const controller = createCanvasOperationController({ isGuardCurrent: () => true });
    const first = controller.start({ cleanupPreview: firstCleanup, guard, identity: filterIdentity })!;
    const publishFirst = vi.fn((_result: string): undefined => undefined);
    const pending = first.run(() => firstWork.promise, publishFirst);
    firstCleanup.mockClear();

    const second = controller.start({
      cleanupPreview: secondCleanup,
      guard: compositeGuard,
      identity: selectObjectIdentity,
    })!;

    expect(firstCleanup).toHaveBeenCalledOnce();
    expect(controller.getSnapshot()).toMatchObject({ identity: selectObjectIdentity, status: 'active' });
    first.cancel();
    firstWork.resolve('old');
    await expect(pending).resolves.toBe('stale');
    expect(publishFirst).not.toHaveBeenCalled();
    expect(secondCleanup).not.toHaveBeenCalled();
    expect(controller.getSnapshot()).toMatchObject({ identity: selectObjectIdentity, status: 'active' });
    second.cancel();
  });

  it('reset aborts work and clears the preview while keeping the operation active', async () => {
    const work = createDeferred<string>();
    const cleanupPreview = vi.fn();
    let signal: AbortSignal | undefined;
    const controller = createCanvasOperationController({ isGuardCurrent: () => true });
    const session = controller.start({ cleanupPreview, guard, identity: filterIdentity })!;
    const pending = session.run(
      (requestSignal) => {
        signal = requestSignal;
        return work.promise;
      },
      vi.fn((): undefined => undefined)
    );
    cleanupPreview.mockClear();

    session.reset();

    expect(signal?.aborted).toBe(true);
    expect(cleanupPreview).toHaveBeenCalledOnce();
    expect(controller.getSnapshot()).toEqual({
      error: null,
      identity: filterIdentity,
      phase: 'ready',
      status: 'active',
    });
    work.resolve('late');
    await expect(pending).resolves.toBe('stale');
  });

  it('cancel aborts work, clears the preview, and closes the operation', async () => {
    const work = createDeferred<string>();
    const cleanupPreview = vi.fn();
    let signal: AbortSignal | undefined;
    const controller = createCanvasOperationController({ isGuardCurrent: () => true });
    const session = controller.start({ cleanupPreview, guard, identity: filterIdentity })!;
    const pending = session.run(
      (requestSignal) => {
        signal = requestSignal;
        return work.promise;
      },
      vi.fn((): undefined => undefined)
    );
    cleanupPreview.mockClear();

    session.cancel();

    expect(signal?.aborted).toBe(true);
    expect(cleanupPreview).toHaveBeenCalledOnce();
    expect(controller.getSnapshot()).toEqual({ status: 'idle' });
    work.resolve('late');
    await expect(pending).resolves.toBe('stale');
  });

  it('dispose aborts work, clears the preview, and stops notifications', async () => {
    const work = createDeferred<string>();
    const cleanupPreview = vi.fn();
    let signal: AbortSignal | undefined;
    const controller = createCanvasOperationController({ isGuardCurrent: () => true });
    const session = controller.start({ cleanupPreview, guard, identity: filterIdentity })!;
    const pending = session.run(
      (requestSignal) => {
        signal = requestSignal;
        return work.promise;
      },
      vi.fn((): undefined => undefined)
    );
    const listener = vi.fn();
    controller.subscribe(listener);
    cleanupPreview.mockClear();
    listener.mockClear();

    controller.dispose();

    expect(signal?.aborted).toBe(true);
    expect(cleanupPreview).toHaveBeenCalledOnce();
    expect(controller.getSnapshot()).toEqual({ status: 'idle' });
    expect(listener).not.toHaveBeenCalled();
    expect(controller.start({ cleanupPreview, guard, identity: filterIdentity })).toBeNull();
    work.resolve('late');
    await expect(pending).resolves.toBe('stale');
  });

  it('publishes only the latest request when completions arrive out of order', async () => {
    const older = createDeferred<string>();
    const newer = createDeferred<string>();
    const publishOlder = vi.fn((_result: string): undefined => undefined);
    const publishNewer = vi.fn((_result: string): undefined => undefined);
    const signals: AbortSignal[] = [];
    const controller = createCanvasOperationController({ isGuardCurrent: () => true });
    const session = controller.start({ cleanupPreview: vi.fn(), guard, identity: filterIdentity })!;

    const olderPending = session.run((signal) => {
      signals.push(signal);
      return older.promise;
    }, publishOlder);
    const newerPending = session.run((signal) => {
      signals.push(signal);
      return newer.promise;
    }, publishNewer);
    newer.resolve('newer');
    await expect(newerPending).resolves.toBe('published');
    older.resolve('older');
    await expect(olderPending).resolves.toBe('stale');

    expect(signals[0]?.aborted).toBe(true);
    expect(publishOlder).not.toHaveBeenCalled();
    expect(publishNewer).toHaveBeenCalledWith('newer');
  });

  it('rechecks freshness after synchronous preview commit', async () => {
    const replacementCleanup = vi.fn();
    const controller = createCanvasOperationController({ isGuardCurrent: () => true });
    const session = controller.start({ cleanupPreview: vi.fn(), guard, identity: filterIdentity })!;

    const result = await session.run(
      () => Promise.resolve('prepared'),
      () => {
        controller.start({ cleanupPreview: replacementCleanup, guard: compositeGuard, identity: selectObjectIdentity });
        return undefined;
      }
    );

    expect(result).toBe('stale');
    expect(controller.getSnapshot()).toMatchObject({ identity: selectObjectIdentity, status: 'active' });
    expect(replacementCleanup).not.toHaveBeenCalled();
  });

  it('requires preview commit callbacks to be synchronous', () => {
    const controller = createCanvasOperationController({ isGuardCurrent: () => true });
    const session = controller.start({ cleanupPreview: vi.fn(), guard, identity: filterIdentity })!;

    if (Date.now() < 0) {
      const asyncCommit = (): Promise<void> => Promise.resolve();
      const voidCommit = (): void => undefined;
      const maybeAsyncCommit = (): void | Promise<void> => Promise.resolve();
      // @ts-expect-error Preview commit must not yield after the final freshness check.
      void session.run(() => Promise.resolve('prepared'), asyncCommit);
      // @ts-expect-error Preview commit must explicitly return undefined.
      void session.run(() => Promise.resolve('prepared'), voidCommit);
      // @ts-expect-error Preview commit cannot have an async branch.
      void session.run(() => Promise.resolve('prepared'), maybeAsyncCommit);
    }
  });

  it('treats AbortError from a superseded request as stale without publishing an error', async () => {
    const controller = createCanvasOperationController({ isGuardCurrent: () => true });
    const session = controller.start({ cleanupPreview: vi.fn(), guard, identity: filterIdentity })!;
    const commitOlder = vi.fn((_result: string): undefined => undefined);
    const older = session.run(
      (signal) =>
        new Promise<string>((_resolve, reject) => {
          signal.addEventListener('abort', () => reject(new DOMException('superseded', 'AbortError')), { once: true });
        }),
      commitOlder
    );

    const commitNewer = vi.fn((_result: string): undefined => undefined);
    const newer = session.run(() => Promise.resolve('newer'), commitNewer);

    await expect(older).resolves.toBe('stale');
    await expect(newer).resolves.toBe('published');
    expect(commitOlder).not.toHaveBeenCalled();
    expect(commitNewer).toHaveBeenCalledWith('newer');
    expect(controller.getSnapshot()).toMatchObject({ error: null, phase: 'ready', status: 'active' });
  });

  it.each(['replacement', 'reset', 'cancel'] as const)(
    'preserves an operation started reentrantly during %s cleanup',
    (lifecycle) => {
      const controller = createCanvasOperationController({ isGuardCurrent: () => true });
      const replacementCleanup = vi.fn();
      const cleanupPreview = vi.fn(() => {
        controller.start({ cleanupPreview: replacementCleanup, guard: compositeGuard, identity: selectObjectIdentity });
      });
      const session = controller.start({ cleanupPreview, guard, identity: filterIdentity })!;

      if (lifecycle === 'replacement') {
        controller.start({ cleanupPreview: vi.fn(), guard, identity: filterIdentity });
      } else if (lifecycle === 'reset') {
        session.reset();
      } else {
        session.cancel();
      }

      expect(cleanupPreview).toHaveBeenCalledOnce();
      expect(replacementCleanup).not.toHaveBeenCalled();
      expect(controller.getSnapshot()).toMatchObject({ identity: selectObjectIdentity, status: 'active' });
    }
  );

  it('preserves an operation started reentrantly during request cleanup', async () => {
    const controller = createCanvasOperationController({ isGuardCurrent: () => true });
    const replacementCleanup = vi.fn();
    const cleanupPreview = vi.fn(() => {
      controller.start({ cleanupPreview: replacementCleanup, guard: compositeGuard, identity: selectObjectIdentity });
    });
    const session = controller.start({ cleanupPreview, guard, identity: filterIdentity })!;
    const work = vi.fn(() => Promise.resolve('prepared'));

    await expect(
      session.run(
        work,
        vi.fn((): undefined => undefined)
      )
    ).resolves.toBe('stale');

    expect(cleanupPreview).toHaveBeenCalledOnce();
    expect(work).not.toHaveBeenCalled();
    expect(replacementCleanup).not.toHaveBeenCalled();
    expect(controller.getSnapshot()).toMatchObject({ identity: selectObjectIdentity, status: 'active' });
  });

  it('contains subscriber exceptions during start and request state transitions', async () => {
    const controller = createCanvasOperationController({ isGuardCurrent: () => true });
    const healthyListener = vi.fn();
    controller.subscribe(() => {
      throw new Error('listener failed');
    });
    controller.subscribe(healthyListener);

    const session = controller.start({ cleanupPreview: vi.fn(), guard, identity: filterIdentity });

    expect(session).not.toBeNull();
    await expect(
      session!.run(
        () => Promise.resolve('ready'),
        vi.fn((): undefined => undefined)
      )
    ).resolves.toBe('published');
    expect(healthyListener).toHaveBeenCalledTimes(3);
    expect(controller.getSnapshot()).toMatchObject({ phase: 'ready', status: 'active' });
  });

  it('rejects an invalid or mismatched export guard without replacing the active operation', () => {
    const isGuardCurrent = vi.fn((candidate: LayerExportGuard) => candidate === guard);
    const controller = createCanvasOperationController({ isGuardCurrent });
    const session = controller.start({ cleanupPreview: vi.fn(), guard, identity: filterIdentity });
    const staleGuard = { ...guard, cacheVersion: 2 };

    expect(session).not.toBeNull();
    expect(controller.start({ cleanupPreview: vi.fn(), guard: staleGuard, identity: filterIdentity })).toBeNull();
    expect(
      controller.start({ cleanupPreview: vi.fn(), guard, identity: { ...filterIdentity, layerId: 'other' } })
    ).toBeNull();
    expect(
      controller.start({ cleanupPreview: vi.fn(), guard, identity: { ...filterIdentity, projectId: 'other' } })
    ).toBeNull();
    expect(controller.getSnapshot()).toMatchObject({ identity: filterIdentity, status: 'active' });
  });

  it('suppresses a result when its guard becomes invalid during work', async () => {
    let current = true;
    const work = createDeferred<string>();
    const publish = vi.fn((_result: string): undefined => undefined);
    const cleanupPreview = vi.fn();
    const controller = createCanvasOperationController({ isGuardCurrent: () => current });
    const session = controller.start({ cleanupPreview, guard, identity: filterIdentity })!;
    const pending = session.run(() => work.promise, publish);
    cleanupPreview.mockClear();

    current = false;
    work.resolve('stale');

    await expect(pending).resolves.toBe('stale');
    expect(publish).not.toHaveBeenCalled();
    expect(cleanupPreview).toHaveBeenCalledOnce();
    expect(controller.getSnapshot()).toEqual({ status: 'idle' });
  });

  it.each([
    [
      'source',
      (controller: ReturnType<typeof createCanvasOperationController>) =>
        controller.invalidateSource('project-1', layer.id),
    ],
    [
      'layer',
      (controller: ReturnType<typeof createCanvasOperationController>) =>
        controller.invalidateLayer('project-1', layer.id),
    ],
    [
      'project',
      (controller: ReturnType<typeof createCanvasOperationController>) => controller.invalidateProject('project-1'),
    ],
    [
      'document',
      (controller: ReturnType<typeof createCanvasOperationController>) => controller.invalidateDocument('project-1'),
    ],
  ])('%s invalidation aborts and cleans the matching operation', async (_reason, invalidate) => {
    const work = createDeferred<string>();
    const cleanupPreview = vi.fn();
    let signal: AbortSignal | undefined;
    const controller = createCanvasOperationController({ isGuardCurrent: () => true });
    const session = controller.start({ cleanupPreview, guard, identity: filterIdentity })!;
    const pending = session.run(
      (requestSignal) => {
        signal = requestSignal;
        return work.promise;
      },
      vi.fn((): undefined => undefined)
    );
    cleanupPreview.mockClear();

    invalidate(controller);

    expect(signal?.aborted).toBe(true);
    expect(cleanupPreview).toHaveBeenCalledOnce();
    expect(controller.getSnapshot()).toEqual({ status: 'idle' });
    work.resolve('late');
    await expect(pending).resolves.toBe('stale');
  });

  it('ignores invalidation for a different target', () => {
    const cleanupPreview = vi.fn();
    const controller = createCanvasOperationController({ isGuardCurrent: () => true });
    controller.start({ cleanupPreview, guard, identity: filterIdentity });

    controller.invalidateSource('project-1', 'other-layer');
    controller.invalidateLayer('other-project', layer.id);
    controller.invalidateProject('other-project');
    controller.invalidateDocument('other-project');

    expect(cleanupPreview).not.toHaveBeenCalled();
    expect(controller.getSnapshot()).toMatchObject({ identity: filterIdentity, status: 'active' });
  });

  it('suppresses a late result after the operation closes', async () => {
    const work = createDeferred<string>();
    const publish = vi.fn((_result: string): undefined => undefined);
    const controller = createCanvasOperationController({ isGuardCurrent: () => true });
    const session = controller.start({
      cleanupPreview: vi.fn(),
      guard: compositeGuard,
      identity: selectObjectIdentity,
    })!;
    const pending = session.run(() => work.promise, publish);

    controller.cancel();
    work.resolve('late');

    await expect(pending).resolves.toBe('stale');
    expect(publish).not.toHaveBeenCalled();
    expect(controller.getSnapshot()).toEqual({ status: 'idle' });
  });

  it('publishes an error and allows the same operation to retry', async () => {
    const cleanupPreview = vi.fn();
    const listener = vi.fn();
    const controller = createCanvasOperationController({ isGuardCurrent: () => true });
    const session = controller.start({ cleanupPreview, guard, identity: filterIdentity })!;
    controller.subscribe(listener);

    await expect(
      session.run(
        () => Promise.reject(new Error('graph failed')),
        vi.fn((): undefined => undefined)
      )
    ).resolves.toBe('error');

    expect(controller.getSnapshot()).toEqual({
      error: 'graph failed',
      identity: filterIdentity,
      phase: 'error',
      status: 'active',
    });

    const publish = vi.fn((_result: string): undefined => undefined);
    await expect(session.run(() => Promise.resolve('recovered'), publish)).resolves.toBe('published');
    expect(publish).toHaveBeenCalledWith('recovered');
    expect(controller.getSnapshot()).toEqual({
      error: null,
      identity: filterIdentity,
      phase: 'ready',
      status: 'active',
    });
    expect(listener).toHaveBeenCalled();
  });
});
