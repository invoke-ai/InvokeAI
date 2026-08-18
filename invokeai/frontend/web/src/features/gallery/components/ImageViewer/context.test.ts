import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

const read = (file: string) => readFileSync(fileURLToPath(new URL(file, import.meta.url)), 'utf8');

// The behaviour of the deferred clear itself is covered by real tests in
// progressImageResolution.test.ts. These are wiring checks only — this directory has no DOM test
// environment, so the provider cannot be mounted. They assert that context.tsx routes through the
// tested unit rather than reimplementing the state inline, which is what previously allowed the
// armed flag and its timer to drift apart.
describe('ImageViewer progress image wiring', () => {
  const context = read('./context.tsx');
  const currentImagePreview = read('./CurrentImagePreview.tsx');

  it('resets the viewer progress atoms on every socket lifecycle transition', () => {
    // socket.io has no event replay, so a drop spanning the terminal queue_item_status_changed
    // loses that event permanently. Without these the overlay covers the finished image until the
    // page is reloaded. setEventListeners already does the same for the global progress stores.
    for (const event of ['connect', 'connect_error', 'disconnect']) {
      expect(context).toContain(`socket.on('${event}', onSocketLifecycleChange)`);
      expect(context).toContain(`socket.off('${event}', onSocketLifecycleChange)`);
    }
  });

  it('keeps the armed flag and its backstop timer in one owned unit', () => {
    // Both must come from createDeferredClear. A local boolean ref plus a separate timeout id is
    // exactly the shape that let a stale timer outlive the generation that armed it.
    expect(context).toContain('createDeferredClear()');
    expect(context).toContain('deferredClear.arm(onResolveDeadline)');
    expect(context).toContain('deferredClear.isArmed()');
    expect(context).not.toContain('shouldClearProgressImageOnLoadRef');
    expect(context).not.toContain('setTimeout');
  });

  it('supersedes a pending backstop when a new progress event arrives', () => {
    // Otherwise: item N completes and arms the backstop, its final image never loads, the user
    // starts item N+1, and N's deadline fires mid-generation and blanks N+1's live preview.
    const progressHandler = context.slice(
      context.indexOf('const onInvocationProgress ='),
      context.indexOf("socket.on('invocation_progress'")
    );
    expect(progressHandler).toContain('disarmDeferredClear()');
  });

  it('promotes a still-active session at the resolve deadline instead of merely disarming', () => {
    // Disarm-only leaves the shared atoms owned by the finished item, so the surviving sessions'
    // terminal events 'ignore' them as foreign and the overlay strands on a stale preview after
    // the last session ends. The promotion policy itself is tested in progressImageResolution.
    const deadlineHandler = context.slice(
      context.indexOf('const onResolveDeadline ='),
      context.indexOf('useEffect(() => {')
    );
    expect(deadlineHandler).toContain('pickPromotionCandidate($activeProgressData.get())');
    expect(deadlineHandler).toContain('$progressEvent.set(candidate.progressEvent)');
    expect(deadlineHandler).toContain('$progressImage.set(candidate.progressImage)');
    expect(deadlineHandler).toContain('disarmDeferredClear()');
    expect(deadlineHandler).toContain('clearProgressImage()');
  });

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
    expect(onReady).toContain('onLoadImage()');
  });
});
