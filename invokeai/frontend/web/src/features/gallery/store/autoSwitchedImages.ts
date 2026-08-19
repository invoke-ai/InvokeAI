/**
 * Marks the gallery selection the auto-switch made, until the viewer renders it.
 *
 * The viewer briefly reveals a newly selected item over the progress overlay so that a
 * mid-generation gallery click is not invisible (see the reveal effect in CurrentImagePreview).
 * Auto-switch selections must not trigger that reveal — but they land asynchronously: the switch is
 * dispatched only after onInvocationComplete's DTO fetch resolves, and the viewer renders it only
 * after the thumbnail preload settles. When the next generation is started quickly, its first
 * invocation_progress event slots into that window and resets $isProgressImageResolving, so by the
 * time the auto-switch selection reaches the viewer it is indistinguishable from a user click and
 * the reveal flashes the previous result over the live preview for 2 seconds.
 *
 * The marker is scoped to the selection it was recorded for, not to the item name: it survives only
 * as long as that selection stands. A name-keyed marker cannot tell "this render is the auto-switch
 * landing" from "the user picked that same item later", so an auto-switch that never rendered —
 * because the user clicked elsewhere first — would swallow their later click on it, the very dead
 * click the reveal exists to prevent. Settling on every selection change closes that: once the
 * selection moves on, the recorded auto-switch can never render, and the marker goes with it.
 *
 * That scoping is also why no expiry or bound is needed. At most one marker exists, and only while
 * its selection is the current one.
 */
type AutoSwitchedSelectionMarker = {
  /** Records that the gallery is auto-switching the selection to this item. Call immediately
   * before dispatching that selection, so the settle it triggers sees the marker. */
  record: (itemName: string) => void;
  /**
   * Points the marker at the selection that now stands, dropping it unless it is still the one
   * recorded. Call on every selection change (see addAutoSwitchedSelectionListener).
   */
  settle: (selectedItemName: string | null) => void;
  /**
   * Returns whether the item now rendering is the one the auto-switch selected, clearing the
   * marker. Call on every change of the rendered item.
   */
  consume: (itemName: string) => boolean;
};

export const createAutoSwitchedSelectionMarker = (): AutoSwitchedSelectionMarker => {
  let pendingItemName: string | null = null;

  return {
    record: (itemName) => {
      pendingItemName = itemName;
    },
    settle: (selectedItemName) => {
      if (pendingItemName !== selectedItemName) {
        pendingItemName = null;
      }
    },
    consume: (itemName) => {
      if (pendingItemName !== itemName) {
        return false;
      }
      pendingItemName = null;
      return true;
    },
  };
};

export const autoSwitchedImages = createAutoSwitchedSelectionMarker();
