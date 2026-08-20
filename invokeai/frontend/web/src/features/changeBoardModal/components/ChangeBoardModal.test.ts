import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { canRetainFailedSelection } from 'features/changeBoardModal/store/slice';
import { describe, expect, it } from 'vitest';

describe('canRetainFailedSelection', () => {
  const unclaimed = { isModalOpen: false, image_names: [], video_names: [] };

  it('allows the write-back when nothing has claimed the modal since', () => {
    expect(canRetainFailedSelection(unclaimed, true)).toBe(true);
  });

  it('refuses once another selection has claimed the modal', () => {
    // Right-click a different image while a large move is in flight: the dialog is open again
    // with that one name in it. Overwriting it with the earlier request's failures would move
    // a set the user never chose, to the board they picked for something else.
    expect(canRetainFailedSelection({ ...unclaimed, isModalOpen: true }, true)).toBe(false);
    expect(canRetainFailedSelection({ ...unclaimed, image_names: ['other.png'] }, true)).toBe(false);
    expect(canRetainFailedSelection({ ...unclaimed, video_names: ['other.mp4'] }, true)).toBe(false);
  });

  it('refuses once the session has ended', () => {
    // The logout listener clears this slice along with the api state; re-seeding it afterwards
    // leaves one user's image names in the next user's store.
    expect(canRetainFailedSelection(unclaimed, false)).toBe(false);
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
});
