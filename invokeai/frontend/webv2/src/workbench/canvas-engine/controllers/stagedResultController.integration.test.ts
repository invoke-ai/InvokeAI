import type { CanvasStagingCandidateContract } from '@workbench/canvas-engine/contracts';
import type { CanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createCanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import { createCanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { describe, expect, it } from 'vitest';

const candidate: CanvasStagingCandidateContract = {
  height: 40,
  imageName: 'rollback-result.png',
  imageUrl: '/rollback-result.png',
  placement: { height: 80, opacity: 0.5, width: 60, x: 12, y: 18 },
  queuedAt: '2026-07-16T00:00:00.000Z',
  sourceQueueItemId: 'queue-rollback',
  thumbnailUrl: '/rollback-result-thumb.png',
  width: 30,
};
const selection = { candidate, selectedImageIndex: 0 } as const;

const createMirrorRejectingPort = (
  store: ReturnType<typeof createWorkbenchStore>,
  projectId: string
): { arm: (type: CanvasProjectMutation['type']) => void; port: CanvasProjectMutationPort } => {
  const real = createCanvasProjectMutationPort(store, projectId);
  let faultActive = false;
  let readsAfterCommit = 0;
  let armedType: CanvasProjectMutation['type'] | null = null;
  const port: CanvasProjectMutationPort = {
    commitEdit: real.commitEdit,
    dispatch: (mutation, origin) => {
      if (mutation.type === armedType) {
        faultActive = true;
        readsAfterCommit = 0;
        armedType = null;
      } else {
        faultActive = false;
      }
      return real.dispatch(mutation, origin);
    },
    getCanvasState: () => {
      if (faultActive) {
        readsAfterCommit += 1;
        if (readsAfterCommit === 1 || readsAfterCommit === 3) {
          throw new Error('document mirror read failed');
        }
      }
      return real.getCanvasState();
    },
    subscribe: real.subscribe,
  };
  return { arm: (type) => (armedType = type), port };
};

describe('staged result project-port integration', () => {
  it('commits the actually selected slot when duplicate candidate keys have different placements', () => {
    const store = createWorkbenchStore();
    const projectId = store.getState().activeProjectId;
    const first = { ...candidate, placement: { ...candidate.placement, x: 10 } };
    const selected = { ...candidate, placement: { ...candidate.placement, x: 90 } };
    store.commands.canvas.appendStagingCandidate({ candidate: first, projectId });
    store.commands.canvas.appendStagingCandidate({ candidate: selected, projectId });
    store.commands.canvas.apply(projectId, { imageIndex: 1, type: 'setStagedImageIndex' });
    const engine = createCanvasEngine({
      ensureProjectOnServer: () => Promise.resolve(),
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      mutationPort: createCanvasProjectMutationPort(store, projectId),
      projectId,
      reportError: () => undefined,
    });

    const result = engine.layers.commitStagedImage({ candidate: selected, selectedImageIndex: 1 });

    expect(result.status).toBe('committed');
    const accepted = store.getState().projects[0]?.canvas.document.stacks.raster[0];
    expect(accepted?.type === 'raster' ? accepted.transform.x : null).toBe(90);
    engine.lifecycle.dispose();
  });

  it('banks a disabled layer without changing the staged session or current layer selection', async () => {
    const store = createWorkbenchStore();
    const projectId = store.getState().activeProjectId;
    const currentLayerId = store.getState().projects[0]!.canvas.document.selectedLayerId;
    store.commands.canvas.appendStagingCandidate({ candidate, projectId });
    const stagedBefore = store.getState().projects[0]!.canvas.stagingArea;
    const engine = createCanvasEngine({
      ensureProjectOnServer: () => Promise.resolve(),
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      mutationPort: createCanvasProjectMutationPort(store, projectId),
      projectId,
      reportError: () => undefined,
    });

    const result = engine.layers.commitStagedImage({ ...selection, continueStaging: true });

    expect(result.status).toBe('committed');
    if (result.status !== 'committed') {
      throw new Error('expected commit');
    }
    const projectAfterSave = store.getState().projects[0]!;
    const savedLayer = projectAfterSave.canvas.document.stacks.raster[0]!;
    expect(savedLayer).toMatchObject({ id: result.layerId, isEnabled: false, type: 'raster' });
    expect(projectAfterSave.canvas.document.selectedLayerId).toBe(currentLayerId);
    expect(projectAfterSave.canvas.stagingArea).toBe(stagedBefore);
    expect(projectAfterSave.canvas.stagingArea.pendingImages).toEqual([candidate]);
    expect(projectAfterSave.canvas.stagingArea.isVisible).toBe(true);

    await engine.history.undo();
    expect(getDocumentLeaves(store.getState().projects[0]!.canvas.document)).not.toContain(savedLayer);
    expect(store.getState().projects[0]!.canvas.stagingArea).toBe(stagedBefore);

    await engine.history.redo();
    expect(store.getState().projects[0]!.canvas.document.stacks.raster[0]).toBe(savedLayer);
    expect(store.getState().projects[0]!.canvas.document.selectedLayerId).toBe(currentLayerId);
    expect(store.getState().projects[0]!.canvas.stagingArea).toBe(stagedBefore);
    engine.lifecycle.dispose();
  });

  it.each([false, true])(
    'rolls back the exact layer, event, selection, and staging when initial mirror acceptance fails (continue: %s)',
    (continueStaging) => {
      const store = createWorkbenchStore();
      const projectId = store.getState().activeProjectId;
      store.commands.canvas.appendStagingCandidate({ candidate, projectId });
      const before = structuredClone(store.getState().projects.find((project) => project.id === projectId)!);
      const rejectingPort = createMirrorRejectingPort(store, projectId);
      rejectingPort.arm('commitStagedImage');
      const engine = createCanvasEngine({
        ensureProjectOnServer: () => Promise.resolve(),
        backend: createTestStubRasterBackend(),
        imageResolver: () => Promise.resolve(new Blob()),
        mutationPort: rejectingPort.port,
        projectId,
        reportError: () => undefined,
      });

      expect(engine.layers.commitStagedImage({ ...selection, continueStaging })).toEqual({
        status: 'stale',
      });
      expect(store.getState().projects.find((project) => project.id === projectId)).toEqual(before);
      expect(engine.document.getDocument()).toEqual(before.canvas.document);
      expect(engine.stores.canUndo.get()).toBe(false);
      engine.lifecycle.dispose();
    }
  );

  it('reconciles the mirror when state reads fail transiently during an undo', async () => {
    const store = createWorkbenchStore();
    const projectId = store.getState().activeProjectId;
    store.commands.canvas.appendStagingCandidate({ candidate, projectId });
    const rejectingPort = createMirrorRejectingPort(store, projectId);
    const engine = createCanvasEngine({
      ensureProjectOnServer: () => Promise.resolve(),
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      mutationPort: rejectingPort.port,
      projectId,
      reportError: () => undefined,
    });
    expect(engine.layers.commitStagedImage(selection).status).toBe('committed');
    const accepted = structuredClone(store.getState().projects.find((project) => project.id === projectId)!);
    rejectingPort.arm('applyCanvasLayerStackMutation');

    expect(await engine.history.undo()).toBe('applied');
    const undone = store.getState().projects.find((project) => project.id === projectId)!;
    expect(undone.canvas.document).not.toEqual(accepted.canvas.document);
    expect(engine.document.getDocument()).toBe(undone.canvas.document);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);
    engine.lifecycle.dispose();
  });

  it('reconciles the mirror when state reads fail transiently during a redo', async () => {
    const store = createWorkbenchStore();
    const projectId = store.getState().activeProjectId;
    store.commands.canvas.appendStagingCandidate({ candidate, projectId });
    const rejectingPort = createMirrorRejectingPort(store, projectId);
    const engine = createCanvasEngine({
      ensureProjectOnServer: () => Promise.resolve(),
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      mutationPort: rejectingPort.port,
      projectId,
      reportError: () => undefined,
    });
    expect(engine.layers.commitStagedImage(selection).status).toBe('committed');
    await engine.history.undo();
    const undone = structuredClone(store.getState().projects.find((project) => project.id === projectId)!);
    rejectingPort.arm('applyCanvasLayerStackMutation');

    expect(await engine.history.redo()).toBe('applied');
    const redone = store.getState().projects.find((project) => project.id === projectId)!;
    expect(redone.canvas.document).not.toEqual(undone.canvas.document);
    expect(engine.document.getDocument()).toBe(redone.canvas.document);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);
    engine.lifecycle.dispose();
  });
});
