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
const selectObjectIdentity = { kind: 'select-object' as const, layerId: layer.id, projectId: 'project-1' };

describe('createCanvasOperationController', () => {
  it('replaces the active operation and prevents its stale session from affecting the replacement', async () => {
    const firstWork = createDeferred<string>();
    const firstCleanup = vi.fn();
    const secondCleanup = vi.fn();
    const controller = createCanvasOperationController({ isGuardCurrent: () => true });
    const first = controller.start({ cleanupPreview: firstCleanup, guard, identity: filterIdentity })!;
    const publishFirst = vi.fn();
    const pending = first.run(() => firstWork.promise, publishFirst);
    firstCleanup.mockClear();

    const second = controller.start({ cleanupPreview: secondCleanup, guard, identity: selectObjectIdentity })!;

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
    const pending = session.run((requestSignal) => {
      signal = requestSignal;
      return work.promise;
    }, vi.fn());
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
    const pending = session.run((requestSignal) => {
      signal = requestSignal;
      return work.promise;
    }, vi.fn());
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
    const pending = session.run((requestSignal) => {
      signal = requestSignal;
      return work.promise;
    }, vi.fn());
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
    const publishOlder = vi.fn();
    const publishNewer = vi.fn();
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
    const publish = vi.fn();
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
    const pending = session.run((requestSignal) => {
      signal = requestSignal;
      return work.promise;
    }, vi.fn());
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
    const publish = vi.fn();
    const controller = createCanvasOperationController({ isGuardCurrent: () => true });
    const session = controller.start({ cleanupPreview: vi.fn(), guard, identity: selectObjectIdentity })!;
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

    await expect(session.run(() => Promise.reject(new Error('graph failed')), vi.fn())).resolves.toBe('error');

    expect(controller.getSnapshot()).toEqual({
      error: 'graph failed',
      identity: filterIdentity,
      phase: 'error',
      status: 'active',
    });

    const publish = vi.fn();
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
