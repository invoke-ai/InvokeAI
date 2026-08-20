import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

const read = (file: string) => readFileSync(fileURLToPath(new URL(file, import.meta.url)), 'utf8');

// Wiring checks only — this directory has no DOM test environment, so the component cannot be
// mounted. The lifecycle behavior behind onLoadImage is covered by real tests in
// viewerProgressLifecycle.test.ts, and the reveal-suppression registry in autoSwitchedImages.test.ts.
describe('CurrentImagePreview reveal wiring', () => {
  const currentImagePreview = read('./CurrentImagePreview.tsx');

  it('gates the viewer reveal on the thumbnail rather than the full-resolution image', () => {
    // Gating on `/full` holds a stale latent preview on screen for the whole multi-megabyte
    // download on a slow connection.
    expect(currentImagePreview).toContain('useMediaUrl(imageDTO?.thumbnail_url)');
    expect(currentImagePreview).toContain('preloader.src = previewSrc');
    expect(currentImagePreview).not.toMatch(/preloader\.src\s*=\s*imageDTO\.image_url/);
  });

  it('clears the progress overlay when the preload settles, including on error', () => {
    // Chakra reports a failed load as onError, not onLoad, so DndImage's onLoad alone is not
    // enough to guarantee the overlay is ever cleared.
    expect(currentImagePreview).toContain('preloader.onerror = onReady');
    const onReady = currentImagePreview.slice(
      currentImagePreview.indexOf('const onReady ='),
      currentImagePreview.indexOf('if (typeof window ===')
    );
    expect(onReady).toContain('onLoadImage(imageDTO.session_id ?? null)');
  });

  it('routes the reveal through the shared machine', () => {
    // Carried over from the #9434 merge, where this checked the controller that preceded the
    // machine: the point is that this component does not reimplement any of the sequencing, so
    // the two media types cannot drift apart and a click that switches type still reads as a
    // selection change. The machine's behavior is unit tested in selectedItemReveal.test.ts, and
    // its connection to the overlay atom is pinned once in CurrentVideoPreview.test.ts, since
    // context.tsx builds a single machine for both previews.
    expect(currentImagePreview).toContain('revealMachine.sync({');
    expect(currentImagePreview).toContain('renderedItemName: imageToRender?.image_name ?? null,');
    // Readiness names the item rather than being a boolean, for the same reason as the video
    // side: a flag reset from a different effect than the one that reads it lifts the overlay
    // onto whatever is on screen.
    expect(currentImagePreview).toContain('isMediaReady: imageToRender !== null,');
    expect(currentImagePreview).toContain('revealMachine.attach()');
    // Nothing of the pre-machine implementations may survive alongside it.
    expect(currentImagePreview).not.toContain('getSelectedItemRevealDecision');
    expect(currentImagePreview).not.toContain('revealController');
    expect(currentImagePreview).not.toContain('autoSwitchedImages');
  });
});
