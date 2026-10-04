import type { GalleryImage } from '@features/gallery';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { WorkbenchState } from '@workbench/projectContracts';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState.testing';
import { beforeEach, describe, expect, it } from 'vitest';

import { createCanvasFromImages } from './createCanvasFromImages';

const image = (imageName: string, width: number, height: number): GalleryImage => ({
  boardId: 'none',
  height,
  imageCategory: 'general',
  imageName,
  imageUrl: `https://images.example/${imageName}`,
  queuedAt: '2026-01-01T00:00:00.000Z',
  sourceQueueItemId: 'queue-1',
  starred: false,
  thumbnailUrl: `https://images.example/thumb/${imageName}`,
  width,
});

/** A tiny store over the real reducer, so project creation and canvas ingestion run for real. */
const createStore = () => {
  let state: WorkbenchState = createInitialWorkbenchState();
  const getProject = (projectId: string) => state.projects.find((project) => project.id === projectId) ?? null;
  return {
    applyCanvasMutation: (projectId: string, mutation: CanvasProjectMutation) => {
      state = workbenchReducer(state, { mutation, projectId, type: 'applyCanvasProjectMutation' });
    },
    createProject: () => {
      state = workbenchReducer(state, { type: 'createProject' });
      return getProject(state.activeProjectId)!;
    },
    getProject,
    getState: () => state,
    isActiveProject: (projectId: string) => state.activeProjectId === projectId,
  };
};

beforeEach(() => {
  accountLifecycle.activate('user-a');
});

describe('createCanvasFromImages', () => {
  it('creates a new active project sized to the largest image and imports every image as a raster layer', async () => {
    const store = createStore();
    const before = store.getState().activeProjectId;

    const result = await createCanvasFromImages({
      ...store,
      images: [image('a.png', 640, 400), image('b.png', 300, 900)],
    });

    expect(result.status).toBe('imported');
    expect(result.projectId).not.toBe(before);
    expect(store.getState().activeProjectId).toBe(result.projectId);
    // Routing to Canvas makes Generate dimensions follow the new frame.
    expect(store.getProject(result.projectId!)!.invocation.sourceId).toBe('canvas');
    const { document } = store.getProject(result.projectId!)!.canvas;
    expect({ height: document.height, width: document.width }).toEqual({ height: 900, width: 640 });
    expect(document.bbox).toEqual({ height: 900, width: 640, x: 0, y: 0 });
    const rasters = getDocumentLeaves(document).filter((leaf) => leaf.type === 'raster');
    expect(
      rasters.map((leaf) => leaf.type === 'raster' && leaf.source.type === 'image' && leaf.source.image.imageName)
    ).toEqual(['a.png', 'b.png']);
    expect(rasters.map((leaf) => [leaf.transform.x, leaf.transform.y])).toEqual([
      [0, 0],
      [0, 0],
    ]);
    expect(document.selectedLayerId).toBe(rasters[1]?.id);
  });

  it('creates nothing for an empty selection', async () => {
    const store = createStore();
    const projects = store.getState().projects.length;

    const result = await createCanvasFromImages({ ...store, images: [] });

    expect(result).toEqual({ projectId: null, status: 'empty' });
    expect(store.getState().projects).toHaveLength(projects);
  });
});
