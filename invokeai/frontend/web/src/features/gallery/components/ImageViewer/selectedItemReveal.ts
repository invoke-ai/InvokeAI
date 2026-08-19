/**
 * Whether the viewer should lift its progress overlay to show the selected item.
 *
 * A generation covers the viewer with an opaque progress overlay, which would otherwise swallow
 * every gallery click for the whole render: the selection changes underneath, but nothing visibly
 * happens. The reveal makes that click land — the clicked item is shown for
 * SELECTED_ITEM_REVEAL_DURATION_MS, then the live preview returns.
 *
 * Only two outcomes, deliberately. The caller clears the running reveal's timer before asking, so a
 * "leave it as it is" answer would strand the reveal on for the rest of the render with no timer
 * left to end it — the decision must always name the state the overlay ends up in.
 */
type SelectedItemRevealDecision = 'reveal' | 'hide';

export type SelectedItemRevealInputs = {
  /** The user's "show progress in viewer" setting. */
  shouldShowProgressInViewer: boolean;
  /** Whether a live progress preview exists to be covering the viewer at all. */
  hasProgressImage: boolean;
  /** Whether a finished generation's preview is mid-handoff to its final image. */
  isProgressImageResolving: boolean;
  /** The item the viewer is rendering right now, or null when it has nothing to show. */
  renderedItemName: string | null;
  /** The item the gallery selection points at. */
  selectedItemName: string | null;
  /** The item the viewer rendered before this change — null on the first render. */
  previousRenderedItemName: string | null;
  /** Whether this render is the gallery auto-switching to a just-finished item, rather than the
   * user picking one. See features/gallery/store/autoSwitchedImages. */
  wasAutoSwitchedTo: boolean;
};

export const getSelectedItemRevealDecision = ({
  shouldShowProgressInViewer,
  hasProgressImage,
  isProgressImageResolving,
  renderedItemName,
  selectedItemName,
  previousRenderedItemName,
  wasAutoSwitchedTo,
}: SelectedItemRevealInputs): SelectedItemRevealDecision => {
  // Nothing is covering the viewer, so there is nothing to reveal from under.
  if (!shouldShowProgressInViewer || !hasProgressImage) {
    return 'hide';
  }

  // The preview is already resolving into a finished generation's final image; a reveal here would
  // fight that handoff.
  if (isProgressImageResolving) {
    return 'hide';
  }

  // Render lagging the selection (the preload has not settled yet): whatever is on screen is not
  // what the user picked, so showing it would not make their click land.
  if (renderedItemName === null || renderedItemName !== selectedItemName) {
    return 'hide';
  }

  // Not a change of displayed item — nothing happened that needs to be made visible. The first
  // render after the viewer opens is not a click either.
  if (previousRenderedItemName === null || previousRenderedItemName === renderedItemName) {
    return 'hide';
  }

  // The reveal exists for *user* selections. An auto-switch to a just-finished item lands
  // asynchronously and can reach the viewer after the next generation's first progress event, so
  // timing cannot tell it apart from a click — treating it as one flashes the finished item over
  // the new generation's live preview for two seconds.
  if (wasAutoSwitchedTo) {
    return 'hide';
  }

  return 'reveal';
};
