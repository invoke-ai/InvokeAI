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
   * Returns whether this image was recently auto-switched to, removing the entry. Call exactly
   * once per rendered-image change — an entry left behind would suppress a genuine user selection
   * of the same image later.
   */
  consume: (imageName: string) => boolean;
};

// A selection can be superseded before it ever renders (rapid back-to-back completions), leaving
// its entry unconsumed. The bound keeps those leftovers from accumulating; a dropped entry's worst
// case is one spurious 2-second reveal.
const MAX_PENDING = 8;

// An entry is only meaningful for the handoff window between the auto-switch dispatch and the
// image's first render (redux propagation plus the thumbnail preload). An entry that outlives that
// window is an orphan — its selection was superseded before rendering, the viewer was unmounted
// (comparison mode), or a duplicate completion event re-recorded an already-rendered image — and
// consuming an orphan later would swallow a genuine user click on that image, the very dead-click
// the reveal exists to prevent. Generous enough for a slow media fetch; expiring early merely
// readmits the 2-second flash on a very slow connection, which is the milder failure.
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
        return false;
      }
      pending.splice(index, 1);
      return true;
    },
  };
};

export const autoSwitchedImages = createAutoSwitchedImageRegistry();
