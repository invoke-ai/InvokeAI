/**
 * Names of images the gallery auto-switched to, pending their first render in the viewer.
 *
 * The viewer briefly reveals a newly selected image over the progress overlay so that a
 * mid-generation gallery click is not invisible (see the reveal effect in CurrentImagePreview).
 * Auto-switch selections must not trigger that reveal — but they land asynchronously: the switch is
 * dispatched only after onInvocationComplete's DTO fetch resolves, and the viewer renders it only
 * after the thumbnail preload settles. When the next generation is started quickly, its first
 * invocation_progress event slots into that window and resets $isProgressImageResolving, so by the
 * time the auto-switch selection reaches the viewer it is indistinguishable from a user click and
 * the reveal flashes the previous result over the live preview for 2 seconds.
 *
 * Recording the image name at dispatch and consuming it on the selection's first render
 * distinguishes the two without depending on event timing.
 */
type AutoSwitchedImageRegistry = {
  /** Records that the gallery is auto-switching to this image. */
  record: (imageName: string) => void;
  /**
   * Returns whether this image was recently auto-switched to. Call exactly once per rendered-image
   * change: every call settles the registry, because each render proves what became of the pending
   * selections. A match removes the entry and every entry recorded before it — those selections
   * were superseded and will never first-render (e.g. two completions within one thumbnail-fetch
   * window; only the last one renders). A miss means an image that was never recorded rendered,
   * i.e. user activity superseded every pending selection, so the registry is cleared entirely.
   * Either way, no entry survives past the next rendered-image change to swallow a genuine user
   * click later — the very dead-click the reveal exists to prevent.
   *
   * The miss-clear is deliberately over-eager: a user click made *before* an entry was recorded
   * can render *after* it (its preload was already in flight), wiping that live entry and
   * readmitting one 2-second flash. That interleave is narrow, and trading it for stale entries
   * that swallow clicks would be backwards — the flash is the milder failure.
   */
  consume: (imageName: string) => boolean;
};

// consume settles the registry on every rendered-image change, so entries can only accumulate
// while nothing renders at all — the viewer unmounted by comparison mode while generations keep
// completing. The bound caps memory there; a dropped entry's worst case is one spurious 2-second
// reveal.
const MAX_PENDING = 8;

// Same no-renders window as above: an entry is only meaningful between the auto-switch dispatch
// and the image's first render, and with the viewer unmounted that render may never come. Without
// the TTL, remounting the viewer within reach of such an orphan and clicking its image would
// suppress the reveal. Generous enough for a slow thumbnail fetch on a bad connection; expiring
// early merely readmits the 2-second flash there, which is the milder failure.
const TTL_MS = 30_000;

export const createAutoSwitchedImageRegistry = (now: () => number = Date.now): AutoSwitchedImageRegistry => {
  let pending: { imageName: string; recordedAt: number }[] = [];

  const prune = () => {
    const cutoff = now() - TTL_MS;
    pending = pending.filter((entry) => entry.recordedAt >= cutoff);
  };

  return {
    record: (imageName) => {
      prune();
      pending.push({ imageName, recordedAt: now() });
      if (pending.length > MAX_PENDING) {
        pending.shift();
      }
    },
    consume: (imageName) => {
      prune();
      const index = pending.findIndex((entry) => entry.imageName === imageName);
      if (index === -1) {
        pending = [];
        return false;
      }
      pending = pending.slice(index + 1);
      return true;
    },
  };
};

export const autoSwitchedImages = createAutoSwitchedImageRegistry();
