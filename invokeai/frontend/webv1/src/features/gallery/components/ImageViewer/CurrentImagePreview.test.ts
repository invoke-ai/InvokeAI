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

  it('routes the reveal through the mounted-tested hook, gated on the settled preload', () => {
    // The wiring is behaviorally covered in useSelectedItemReveal.test.tsx; this pins that the
    // component uses it, and that readiness means "the preload settled" on the image path.
    expect(currentImagePreview).toContain('useSelectedItemReveal({');
    expect(currentImagePreview).toContain('isMediaReady: imageToRender !== null,');
  });
});
