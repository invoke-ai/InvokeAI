import { atom } from 'nanostores';

/**
 * Who made the gallery's current selection, and which selection it is.
 *
 * The viewer briefly reveals a newly selected item over an in-progress generation's opaque overlay,
 * so that a mid-render gallery click is not invisible. Deciding whether to do that needs two facts
 * the selection itself does not carry:
 *
 * - **Who selected it.** The gallery auto-switches to a finished item, and that dispatch lands
 *   after an async DTO fetch — late enough that the next generation's first progress event can slot
 *   in ahead of it. Timing therefore cannot tell the handoff from a click; the source has to be
 *   recorded at the dispatch that makes it.
 * - **Which selection it is.** "The user picked X" and "X has been the selection for a while" are
 *   different events, and re-picking the item already on screen is a third. A monotonic generation
 *   distinguishes all three without keeping a previous-item name around to compare against — the
 *   arrangement this replaces, which needed a shared mutable ref between the two preview components
 *   and a sentinel value to represent "the selection was cleared".
 */
export type GallerySelectionDescriptor = {
  /** The active item (the last of the selection), or null when nothing is selected. */
  name: string | null;
  /** Incremented for every selection dispatch, including re-selecting the item already active. */
  generation: number;
  /** True when the gallery's auto-switch made this selection rather than the user. */
  isAutoSwitch: boolean;
};

export const $gallerySelection = atom<GallerySelectionDescriptor>({
  name: null,
  generation: 0,
  isAutoSwitch: false,
});

// Set immediately before an auto-switch dispatches its selection, consumed by the very next
// recorded selection. The dispatch follows synchronously, so nothing can slip in between.
let nextSelectionIsAutoSwitch = false;

/** Marks the selection an auto-switch is about to dispatch. See onInvocationComplete. */
export const markNextSelectionAutoSwitched = () => {
  nextSelectionIsAutoSwitch = true;
};

/** Records the selection that just landed. Called by addGallerySelectionSourceListener. */
export const recordGallerySelection = (name: string | null) => {
  const isAutoSwitch = nextSelectionIsAutoSwitch;
  nextSelectionIsAutoSwitch = false;
  const previous = $gallerySelection.get();
  $gallerySelection.set({ name, generation: previous.generation + 1, isAutoSwitch });
};

/** Test seam: drops the pending auto-switch mark and resets the descriptor. */
export const resetGallerySelectionSource = () => {
  nextSelectionIsAutoSwitch = false;
  $gallerySelection.set({ name: null, generation: 0, isAutoSwitch: false });
};
