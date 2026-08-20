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

  it('routes the reveal through the shared controller with the auto-switch marker', () => {
    // Mirrors the same check on CurrentVideoPreview: both previews must go through one controller
    // or the two media types drift apart, and a click that switches type stops reading as a
    // selection change. The controller owns the sequencing the component used to inline -- marker
    // consumption, resolve-window deferral, StrictMode re-arm -- and its branches are unit tested
    // in selectedItemReveal.test.ts, including that the marker is consumed on every change of the
    // rendered item even when no progress is showing.
    expect(currentImagePreview).toMatch(/marker: autoSwitchedImages,/);
    expect(currentImagePreview).toMatch(/revealController\.run\(\{/);
    expect(currentImagePreview).toMatch(/isProgressImageResolving,\s+renderedItemName: imageToRender\?\.image_name/);
    // The component must not reimplement any of it alongside the controller.
    expect(currentImagePreview).not.toContain('getSelectedItemRevealDecision');
    expect(currentImagePreview).not.toContain('autoSwitchedImages.consume(');
    // The effect cleanup only cancels the timer; the next run owns the revealed flag.
    expect(currentImagePreview).toMatch(/return \(\) => \{\s+revealController\.clearTimer\(\);\s+\};/);
  });
});
