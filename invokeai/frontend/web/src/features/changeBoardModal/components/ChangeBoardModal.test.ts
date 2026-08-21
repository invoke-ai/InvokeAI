import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import {
  canRetainFailedSelection,
  changeBoardModalSliceConfig,
  changeBoardOperationInvalidated,
  changeBoardReset,
  imagesToChangeSelected,
} from 'features/changeBoardModal/store/slice';
import { describe, expect, it } from 'vitest';

describe('canRetainFailedSelection', () => {
  const unclaimed = { operation_id: 3, isModalOpen: false, image_names: [], video_names: [] };

  it('allows the write-back when nothing has claimed the modal since', () => {
    expect(canRetainFailedSelection(unclaimed, 3, true)).toBe(true);
  });

  it('refuses once another selection has claimed the modal', () => {
    // Right-click a different image while a large move is in flight: the dialog is open again
    // with that one name in it. Overwriting it with the earlier request's failures would move
    // a set the user never chose, to the board they picked for something else.
    expect(canRetainFailedSelection({ ...unclaimed, isModalOpen: true }, 3, true)).toBe(false);
    expect(canRetainFailedSelection({ ...unclaimed, image_names: ['other.png'] }, 3, true)).toBe(false);
    expect(canRetainFailedSelection({ ...unclaimed, video_names: ['other.mp4'] }, 3, true)).toBe(false);
  });

  it('refuses after a newer selection was opened and canceled', () => {
    expect(canRetainFailedSelection(unclaimed, 4, true)).toBe(false);
  });

  it('refuses once the session has ended', () => {
    // The logout listener clears this slice along with the api state; re-seeding it afterwards
    // leaves one user's image names in the next user's store.
    expect(canRetainFailedSelection(unclaimed, 3, false)).toBe(false);
  });
});

describe('change board operation ownership', () => {
  it('advances on selection but preserves ownership across the accept reset', () => {
    const reducer = changeBoardModalSliceConfig.slice.reducer;
    const selected = reducer(undefined, imagesToChangeSelected(['first.png']));
    const reset = reducer(selected, changeBoardReset());
    const newer = reducer(reset, imagesToChangeSelected(['second.png']));
    const invalidated = reducer(newer, changeBoardOperationInvalidated());

    expect(selected.operation_id).toBe(1);
    expect(reset.operation_id).toBe(1);
    expect(newer.operation_id).toBe(2);
    expect(invalidated.operation_id).toBe(3);
    expect(invalidated.image_names).toEqual([]);
  });
});

/**
 * A source-level guard, in the manner of videoReviewRegressions: this repo does not do UI
 * tests, and what is being guarded here is an ordering property rather than a value. Accepting
 * the dialog resets the selection on the way out (ConfirmationAlertDialog calls acceptCallback
 * and then onClose), so an image mutation that is fired and forgotten has its `failed_images`
 * cleared away with everything else — the images that did not move leave no trace to retry from.
 */
describe('ChangeBoardModal', () => {
  const source = readFileSync(fileURLToPath(new URL('./ChangeBoardModal.tsx', import.meta.url)), 'utf8');

  it('awaits the image board mutation instead of firing and forgetting it', () => {
    expect(source).toMatch(/addImagesToBoard\([\s\S]*?\.unwrap\(\)/);
    expect(source).toContain('result.failed_images');
    expect(source).toMatch(/failedImageNames\.length === 0[\s\S]*changeBoardReset/);
    // Every late write goes through the guard, the reset included — it can clear a selection
    // that now belongs to someone else just as easily as the retain can overwrite one.
    expect(source).toMatch(/canRetainFailedSelection\([\s\S]*?return;[\s\S]*changeBoardReset/);
  });

  it('keeps the images that did not move selected', () => {
    expect(source).toContain('imagesToChangeSelected(failedImageNames)');
    // A rejected request moved nothing at all, so the whole request stays selected.
    expect(source).toContain('.catch(() => imagesToChange)');
  });

  it('captures the operation id before awaiting the move, not after it', () => {
    // The whole ownership check rests on this ordering. Read after the await, the id is
    // whatever the slice holds once every newer selection has already come and gone, so the
    // guard compares the current value against itself and admits every stale operation --
    // silently, since it still typechecks and every other assertion here still passes.
    const capture = source.indexOf('const operationId =');
    const settle = source.indexOf('await Promise.all');

    expect(capture).toBeGreaterThan(-1);
    expect(settle).toBeGreaterThan(-1);
    expect(capture).toBeLessThan(settle);
  });

  it('reports failed video moves ahead of the ownership guard', () => {
    // This toast is the only failure report the video board routes have: no onQueryStarted, no
    // matchRejected listener, unlike the image batch routes. Behind the guard, opening and
    // cancelling any second dialog while the move was in flight leaves the user with no notice
    // at all that it failed.
    const videoToast = source.indexOf('VIDEOS_FAILED_TO_MOVE');
    const guard = source.indexOf('canRetainFailedSelection(');

    expect(videoToast).toBeGreaterThan(-1);
    expect(guard).toBeGreaterThan(-1);
    expect(videoToast).toBeLessThan(guard);
  });
});
