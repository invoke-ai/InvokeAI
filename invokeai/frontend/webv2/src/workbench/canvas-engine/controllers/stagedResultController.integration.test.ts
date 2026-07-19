import type { CanvasStagingCandidateContract } from '@workbench/canvas-engine/contracts';
import type { CanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

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
    dispatch: (mutation) => {
      if (mutation.type === armedType) {
        faultActive = true;
        readsAfterCommit = 0;
        armedType = null;
      } else {
        faultActive = false;
      }
      return real.dispatch(mutation);
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
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      mutationPort: createCanvasProjectMutationPort(store, projectId),
      projectId,
      reportError: () => undefined,
    });

    const result = engine.layers.commitStagedImage({ candidate: selected, selectedImageIndex: 1 });

    expect(result.status).toBe('committed');
    expect(store.getState().projects[0]?.canvas.document.layers[0]?.transform.x).toBe(90);
    engine.lifecycle.dispose();
  });

  it('rolls back the exact layer, event, selection, and staging when initial mirror acceptance fails', () => {
    const store = createWorkbenchStore();
    const projectId = store.getState().activeProjectId;
    store.commands.canvas.appendStagingCandidate({ candidate, projectId });
    const before = structuredClone(store.getState().projects.find((project) => project.id === projectId)!);
    const rejectingPort = createMirrorRejectingPort(store, projectId);
    rejectingPort.arm('commitStagedImage');
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      mutationPort: rejectingPort.port,
      projectId,
      reportError: () => undefined,
    });

    expect(engine.layers.commitStagedImage(selection)).toEqual({
      status: 'stale',
    });
    expect(store.getState().projects.find((project) => project.id === projectId)).toEqual(before);
    expect(engine.document.getDocument()).toEqual(before.canvas.document);
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('compensates a reducer-accepted undo when mirror acceptance fails and keeps history retryable', () => {
    const store = createWorkbenchStore();
    const projectId = store.getState().activeProjectId;
    store.commands.canvas.appendStagingCandidate({ candidate, projectId });
    const rejectingPort = createMirrorRejectingPort(store, projectId);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      mutationPort: rejectingPort.port,
      projectId,
      reportError: () => undefined,
    });
    expect(engine.layers.commitStagedImage(selection).status).toBe('committed');
    const accepted = structuredClone(store.getState().projects.find((project) => project.id === projectId)!);
    rejectingPort.arm('applyCanvasLayerStackMutation');

    expect(() => engine.history.undo()).toThrow('document mirror read failed');
    expect(store.getState().projects.find((project) => project.id === projectId)).toEqual(accepted);
    expect(engine.document.getDocument()).toEqual(accepted.canvas.document);
    expect(engine.stores.canUndo.get()).toBe(true);
    expect(engine.stores.canRedo.get()).toBe(false);
    engine.lifecycle.dispose();
  });

  it('compensates a reducer-accepted redo when mirror acceptance fails and keeps history retryable', () => {
    const store = createWorkbenchStore();
    const projectId = store.getState().activeProjectId;
    store.commands.canvas.appendStagingCandidate({ candidate, projectId });
    const rejectingPort = createMirrorRejectingPort(store, projectId);
    const engine = createCanvasEngine({
      backend: createTestStubRasterBackend(),
      imageResolver: () => Promise.resolve(new Blob()),
      mutationPort: rejectingPort.port,
      projectId,
      reportError: () => undefined,
    });
    expect(engine.layers.commitStagedImage(selection).status).toBe('committed');
    engine.history.undo();
    const undone = structuredClone(store.getState().projects.find((project) => project.id === projectId)!);
    rejectingPort.arm('applyCanvasLayerStackMutation');

    expect(() => engine.history.redo()).toThrow('document mirror read failed');
    expect(store.getState().projects.find((project) => project.id === projectId)).toEqual(undone);
    expect(engine.document.getDocument()).toEqual(undone.canvas.document);
    expect(engine.stores.canUndo.get()).toBe(false);
    expect(engine.stores.canRedo.get()).toBe(true);
    engine.lifecycle.dispose();
  });
});
