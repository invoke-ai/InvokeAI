import type { CanvasStagingCandidateContract } from '@workbench/canvas-engine/api';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchQueueItem as QueueItem } from '@workbench/queueHistoryContracts';

import { getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createCanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import { createCanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import { getCanvasStagingSlots } from '@workbench/canvasStagingView';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { describe, expect, it } from 'vitest';

import { acceptStagedCandidate, getStoppableCandidateBatch } from './stagedAcceptance';

const image = (imageName: string, sourceQueueItemId: string) => ({
  height: 64,
  imageName,
  imageUrl: `/${imageName}`,
  queuedAt: '2026-07-16T00:00:00.000Z',
  sourceQueueItemId,
  thumbnailUrl: `/${imageName}.thumb`,
  width: 64,
});

const candidate: CanvasStagingCandidateContract = {
  ...image('left-eye.png', 'batch-eyes'),
  placement: { height: 64, opacity: 1, width: 64, x: 0, y: 0 },
  sourceBackendItemId: 1,
};

const queueItem = (
  project: Project,
  id: string,
  backendItemIds: number[],
  overrides: Partial<QueueItem> = {}
): QueueItem =>
  ({
    backendBatchId: `${id}-backend`,
    backendItemIds,
    cancellable: true,
    id,
    snapshot: {
      canvas: { document: project.canvas.document, documentRevision: project.canvas.documentRevision },
      destination: 'canvas',
      presentation: { batchCount: backendItemIds.length, height: 64, width: 64 },
      sourceId: 'canvas',
      submittedAt: '2026-07-16T00:00:00.000Z',
    },
    status: 'running',
    ...overrides,
  }) as QueueItem;

/** A project staging the first image of a three-image batch, beside an unrelated running batch. */
const setup = (eyes: Partial<QueueItem> = {}) => {
  const initial = createInitialWorkbenchState();
  const base = initial.projects[0]!;
  const store = createWorkbenchStore({
    ...initial,
    projects: [
      {
        ...base,
        queue: {
          items: [
            queueItem(base, 'batch-eyes', [1, 2, 3], { completedBackendItemIds: [1], ...eyes }),
            queueItem(base, 'batch-other', [10]),
          ],
        },
      },
    ],
  });
  const projectId = store.getState().activeProjectId;
  store.commands.canvas.appendStagingCandidate({ candidate, projectId });
  store.commands.canvas.apply(projectId, { imageIndex: 0, type: 'setStagedImageIndex' });
  const engine = createCanvasEngine({
    backend: createTestStubRasterBackend(),
    ensureProjectOnServer: () => Promise.resolve(),
    imageResolver: () => Promise.resolve(new Blob()),
    mutationPort: createCanvasProjectMutationPort(store, projectId),
    projectId,
    reportError: () => undefined,
  });
  const project = () => store.getState().projects.find((entry) => entry.id === projectId)!;
  const stopped: string[] = [];
  const accept = (continueStaging = false, selectedImageIndex = 0) =>
    acceptStagedCandidate(
      {
        commit: (options) => engine.layers.commitStagedImage(options),
        stopBatch: (queueItemId) => {
          stopped.push(queueItemId);
          store.commands.queue.cancel(projectId, queueItemId);
        },
      },
      { candidate, continueStaging, queueItems: project().queue.items, selectedImageIndex }
    );
  const status = (id: string) => project().queue.items.find((item) => item.id === id)?.status;
  return { accept, engine, project, projectId, status, stopped, store };
};

describe('getStoppableCandidateBatch', () => {
  const project = createInitialWorkbenchState().projects[0]!;

  it.each([
    ['a running batch', { status: 'running' }, 'batch-eyes'],
    ['a queued batch', { status: 'pending' }, 'batch-eyes'],
    ['a finished batch', { status: 'completed' }, null],
    ['an already cancelled batch', { status: 'cancelled' }, null],
    ['a batch the backend cannot cancel', { cancellable: false, status: 'running' }, null],
  ] as const)('names %s accordingly', (_label, overrides, expected) => {
    expect(getStoppableCandidateBatch(candidate, [queueItem(project, 'batch-eyes', [1, 2], overrides)])).toBe(expected);
  });

  it('ignores other batches and a candidate whose batch is gone', () => {
    expect(getStoppableCandidateBatch(candidate, [queueItem(project, 'batch-other', [10])])).toBeNull();
  });
});

describe('acceptStagedCandidate', () => {
  it('accepts, then stops only the candidate batch, so its late siblings never reopen staging', () => {
    const h = setup();
    expect(getCanvasStagingSlots(h.project().canvas, h.project().queue.items)[0]).toMatchObject({
      candidate,
      kind: 'candidate',
    });

    const result = h.accept();

    expect(result.status).toBe('committed');
    expect(h.stopped).toEqual(['batch-eyes']);
    expect(h.project().queue.items.find((item) => item.id === 'batch-eyes')).toMatchObject({
      cancellationPending: true,
      status: 'cancelled',
    });
    expect(h.status('batch-other')).toBe('running');
    const layerId = result.status === 'committed' ? result.layerId : '';
    expect(h.project().canvas.document.stacks.raster[0]).toMatchObject({ id: layerId, isEnabled: true });

    // A sibling that finished before the backend stopped is dropped; the unrelated batch still stages.
    h.store.commands.queue.routePartialResults({
      backendItemId: 2,
      images: [image('right-eye.png', 'batch-eyes')],
      projectId: h.projectId,
      queueItemId: 'batch-eyes',
      videos: [],
    });
    expect(h.project().canvas.stagingArea.pendingImages).toEqual([]);
    h.store.commands.queue.routePartialResults({
      backendItemId: 10,
      images: [image('other.png', 'batch-other')],
      projectId: h.projectId,
      queueItemId: 'batch-other',
      videos: [],
    });
    expect(h.project().canvas.stagingArea.pendingImages.map((pending) => pending.imageName)).toEqual(['other.png']);
    h.engine.lifecycle.dispose();
  });

  it('keeps the batch stopped when the accepted layer is undone', async () => {
    const h = setup();
    const result = h.accept();
    const layerId = result.status === 'committed' ? result.layerId : '';
    const queueAfterAccept = h.project().queue;

    expect(await h.engine.history.undo()).toBe('applied');

    expect(getDocumentLeaves(h.project().canvas.document).some((leaf) => leaf.id === layerId)).toBe(false);
    expect(h.project().queue).toBe(queueAfterAccept);
    expect(h.status('batch-eyes')).toBe('cancelled');
    h.engine.lifecycle.dispose();
  });

  it('leaves the batch running when the accept is refused', () => {
    const h = setup();

    // Index 1 is a placeholder, not the candidate the user saw.
    expect(h.accept(false, 1).status).not.toBe('committed');

    expect(h.stopped).toEqual([]);
    expect(h.status('batch-eyes')).toBe('running');
    h.engine.lifecycle.dispose();
  });

  it('keeps generating when the result is kept as a hidden layer to continue comparing', () => {
    const h = setup();

    expect(h.accept(true).status).toBe('committed');

    expect(h.stopped).toEqual([]);
    expect(h.status('batch-eyes')).toBe('running');
    expect(h.project().canvas.stagingArea.pendingImages).toEqual([candidate]);
    h.engine.lifecycle.dispose();
  });

  it('accepts without stopping anything once the batch has finished', () => {
    const h = setup({ completedBackendItemIds: [1, 2, 3], status: 'completed' });

    expect(h.accept().status).toBe('committed');

    expect(h.stopped).toEqual([]);
    expect(h.status('batch-eyes')).toBe('completed');
    expect(h.status('batch-other')).toBe('running');
    h.engine.lifecycle.dispose();
  });
});
