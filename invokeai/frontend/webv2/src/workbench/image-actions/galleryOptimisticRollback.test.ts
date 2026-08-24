import type { GalleryItem } from '@features/gallery';
import type { Project, WorkbenchState } from '@workbench/projectContracts';

import { getProjectWidgetValues } from '@workbench/widgetState';
import { createInitialWorkbenchState, workbenchReducer, type WorkbenchAction } from '@workbench/workbenchState.testing';
import { describe, expect, it } from 'vitest';

import {
  captureGalleryWidgetKeyValues,
  diffGalleryWidgetKeyValues,
  selectRestorableGalleryWidgetPatches,
  type GalleryWidgetRestorePatch,
} from './galleryOptimisticRollback';

/**
 * Reducer-level coverage for the delete-rollback widget-store restore
 * mechanism (Task 7 review finding #6): exercises the real `workbenchState`
 * reducer end to end, rather than a mocked store, so the capture/diff/select
 * functions are proven against the actual `removeGalleryItems` and
 * `patchWidgetValues` contracts instead of a hand-rolled stand-in that could
 * silently drift from them.
 */

const galleryItem: GalleryItem = {
  boardId: 'none',
  category: 'general',
  createdAt: '2026-07-30T00:00:00.000Z',
  fullUrl: '/api/v1/images/i/a.png/full',
  height: 512,
  isIntermediate: false,
  kind: 'image',
  name: 'a.png',
  starred: false,
  thumbnailUrl: '/api/v1/images/i/a.png/thumbnail',
  width: 512,
};

const recentImage = {
  boardId: 'none',
  height: 512,
  imageName: 'a.png',
  imageUrl: galleryItem.fullUrl,
  queuedAt: galleryItem.createdAt,
  sourceQueueItemId: 'backend-gallery',
  thumbnailUrl: galleryItem.thumbnailUrl,
  width: 512,
};

const upscaleInputImage = { height: 512, image_name: 'a.png', width: 512 };

const getProject = (state: WorkbenchState, projectId: string): Project => {
  const project = state.projects.find((candidate) => candidate.id === projectId);

  expect(project).toBeDefined();

  return project as Project;
};

/** Seeds a fresh project with `a.png` selected, compared, in the recent-image
 *  overlay, and locked as the upscale widget's input — everything
 *  `removeGalleryItemsFromAllProjects` can touch. */
const seedProjectWithItem = (): { projectId: string; state: WorkbenchState } => {
  let state = createInitialWorkbenchState();
  const projectId = state.activeProjectId;

  state = workbenchReducer(state, {
    projectId,
    type: 'patchWidgetValues',
    values: {
      compareImage: galleryItem,
      recentImages: [recentImage],
      selectedImage: galleryItem,
      selectedImageName: 'image:a.png',
      selectedImageNames: ['image:a.png'],
    },
    widgetId: 'gallery',
  });
  state = workbenchReducer(state, {
    projectId,
    type: 'patchWidgetValues',
    values: { inputImage: upscaleInputImage },
    widgetId: 'upscale',
  });

  return { projectId, state };
};

const applyRestorePatches = (state: WorkbenchState, patches: GalleryWidgetRestorePatch[]): WorkbenchState =>
  patches.reduce<WorkbenchState>(
    (nextState, patch) =>
      workbenchReducer(nextState, {
        projectId: patch.projectId,
        type: 'patchWidgetValues',
        values: patch.values,
        widgetId: patch.widgetId,
      } as WorkbenchAction),
    state
  );

describe('gallery optimistic rollback against the real reducer', () => {
  it('restores recentImages, selection, compareImage, and upscale.inputImage after a totally-failed delete', () => {
    const { projectId, state } = seedProjectWithItem();

    const before = captureGalleryWidgetKeyValues(state.projects);
    const afterRemoval = workbenchReducer(state, { itemKeys: ['image:a.png'], type: 'removeGalleryItems' });

    // Sanity: the removal really did clear everything we're about to assert
    // gets restored, otherwise this test would pass trivially.
    const clearedValues = getProjectWidgetValues(getProject(afterRemoval, projectId), 'gallery');

    expect(clearedValues).toMatchObject({
      compareImage: null,
      recentImages: [],
      selectedImage: null,
      selectedImageName: null,
      selectedImageNames: [],
    });
    expect(getProjectWidgetValues(getProject(afterRemoval, projectId), 'upscale').inputImage).toBeNull();

    const entries = diffGalleryWidgetKeyValues(before, afterRemoval.projects);

    expect(entries.length).toBeGreaterThan(0);

    const patches = selectRestorableGalleryWidgetPatches(entries, afterRemoval.projects);
    const restored = applyRestorePatches(afterRemoval, patches);

    const restoredGalleryValues = getProjectWidgetValues(getProject(restored, projectId), 'gallery');

    expect(restoredGalleryValues).toMatchObject({
      compareImage: galleryItem,
      recentImages: [recentImage],
      selectedImage: galleryItem,
      selectedImageName: 'image:a.png',
      selectedImageNames: ['image:a.png'],
    });
    expect(getProjectWidgetValues(getProject(restored, projectId), 'upscale').inputImage).toEqual(upscaleInputImage);
  });

  it('does not clobber a field a concurrent mutation changed after the diff was captured', () => {
    const { projectId, state } = seedProjectWithItem();

    const before = captureGalleryWidgetKeyValues(state.projects);
    const afterRemoval = workbenchReducer(state, { itemKeys: ['image:a.png'], type: 'removeGalleryItems' });
    const entries = diffGalleryWidgetKeyValues(before, afterRemoval.projects);

    // A new generation completes mid-flight and rewrites `recentImages`
    // before the rollback runs. Nothing else changes.
    const concurrentImage = { ...recentImage, imageName: 'new-generation.png' };
    const concurrentlyChanged = workbenchReducer(afterRemoval, {
      projectId,
      type: 'patchWidgetValues',
      values: { recentImages: [concurrentImage] },
      widgetId: 'gallery',
    });

    const patches = selectRestorableGalleryWidgetPatches(entries, concurrentlyChanged.projects);
    const restored = applyRestorePatches(concurrentlyChanged, patches);

    const restoredGalleryValues = getProjectWidgetValues(getProject(restored, projectId), 'gallery');

    // The concurrent write to `recentImages` survives untouched...
    expect(restoredGalleryValues.recentImages).toEqual([concurrentImage]);
    // ...while every field nothing else touched still restores correctly.
    expect(restoredGalleryValues.compareImage).toEqual(galleryItem);
    expect(restoredGalleryValues.selectedImage).toEqual(galleryItem);
    expect(restoredGalleryValues.selectedImageName).toBe('image:a.png');
    expect(restoredGalleryValues.selectedImageNames).toEqual(['image:a.png']);
    expect(getProjectWidgetValues(getProject(restored, projectId), 'upscale').inputImage).toEqual(upscaleInputImage);
  });

  it('does not restore a video slot whose exclusive rival was claimed while the delete was in flight', () => {
    const seeded = seedProjectWithItem();
    const projectId = seeded.projectId;
    const firstFrame = { height: 480, image_name: 'a.png', width: 832 };
    const sourceClip = {
      endFrame: 79,
      fps: 16,
      height: 480,
      numFrames: 81,
      startFrame: 0,
      video_name: 'clip.mp4',
      width: 832,
    };
    const state = workbenchReducer(seeded.state, {
      projectId,
      type: 'patchWidgetValues',
      values: { firstFrameImage: firstFrame },
      widgetId: 'video',
    });

    const before = captureGalleryWidgetKeyValues(state.projects);
    const afterRemoval = workbenchReducer(state, { itemKeys: ['image:a.png'], type: 'removeGalleryItems' });

    // Sanity: the optimistic removal cleared the video slot.
    expect(getProjectWidgetValues(getProject(afterRemoval, projectId), 'video').firstFrameImage).toBeNull();

    const entries = diffGalleryWidgetKeyValues(before, afterRemoval.projects);

    // The slot freed up, so the user drops an initial video — exactly the
    // patch `setSourceVideo` makes — before the server delete fails.
    const claimed = workbenchReducer(afterRemoval, {
      projectId,
      type: 'patchWidgetValues',
      values: { firstFrameImage: null, sourceVideo: sourceClip },
      widgetId: 'video',
    });

    const patches = selectRestorableGalleryWidgetPatches(entries, claimed.projects);
    const restored = applyRestorePatches(claimed, patches);
    const videoValues = getProjectWidgetValues(getProject(restored, projectId), 'video');

    // Restoring the first frame would recreate the forbidden
    // first-frame+initial-video pair; the restore loses.
    expect(videoValues.firstFrameImage).toBeNull();
    expect(videoValues.sourceVideo).toEqual(sourceClip);
    // Unrelated fields still restore normally.
    expect(getProjectWidgetValues(getProject(restored, projectId), 'gallery').selectedImage).toEqual(galleryItem);
  });

  it('still restores a cleared video slot when its rival stayed empty', () => {
    const seeded = seedProjectWithItem();
    const projectId = seeded.projectId;
    const firstFrame = { height: 480, image_name: 'a.png', width: 832 };
    const state = workbenchReducer(seeded.state, {
      projectId,
      type: 'patchWidgetValues',
      values: { firstFrameImage: firstFrame },
      widgetId: 'video',
    });

    const before = captureGalleryWidgetKeyValues(state.projects);
    const afterRemoval = workbenchReducer(state, { itemKeys: ['image:a.png'], type: 'removeGalleryItems' });
    const entries = diffGalleryWidgetKeyValues(before, afterRemoval.projects);
    const patches = selectRestorableGalleryWidgetPatches(entries, afterRemoval.projects);
    const restored = applyRestorePatches(afterRemoval, patches);

    expect(getProjectWidgetValues(getProject(restored, projectId), 'video').firstFrameImage).toEqual(firstFrame);
  });

  it('restores nothing when the optimistic removal did not actually change anything', () => {
    const { state } = seedProjectWithItem();
    const before = captureGalleryWidgetKeyValues(state.projects);

    // Removing an unrelated key changes nothing in this project.
    const afterRemoval = workbenchReducer(state, { itemKeys: ['image:unrelated.png'], type: 'removeGalleryItems' });
    const entries = diffGalleryWidgetKeyValues(before, afterRemoval.projects);

    expect(entries).toEqual([]);
    expect(selectRestorableGalleryWidgetPatches(entries, afterRemoval.projects)).toEqual([]);
  });
});
