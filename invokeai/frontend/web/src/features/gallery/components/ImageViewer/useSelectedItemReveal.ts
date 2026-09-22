import type { WritableAtom } from 'nanostores';
import { useCallback, useEffect, useState } from 'react';

import { createSelectedItemRevealController } from './selectedItemReveal';

type UseSelectedItemRevealArgs = {
  /** Shared with the other preview component via the viewer context, so a click that switches
   * media type still reads as a selection change on both ends. */
  lastRenderedItemNameRef: { current: string | null };
  /** The shared overlay-lift flag the progress overlay reads. */
  $isTemporarilyShowingSelectedImage: WritableAtom<boolean>;
  /** The auto-switch marker (see features/gallery/store/autoSwitchedImages). */
  marker: { consume: (itemName: string) => boolean };
  durationMs: number;
  mediaGraceMs: number;
  /** The item the component is rendering right now, or null when it has nothing to show. */
  renderedItemName: string | null;
  /** Whether that item has a frame on screen. See usePaintedItemName. */
  isMediaReady: boolean;
  /** The item the gallery selection points at, or null when the selection is empty. */
  selectedItemName: string | null;
  shouldShowProgressInViewer: boolean;
  hasProgressImage: boolean;
  isProgressImageResolving: boolean;
};

/**
 * The preview components' reveal wiring, extracted so it can be mounted and tested with real
 * effect lifecycles (see useSelectedItemReveal.test.tsx). The sequencing itself lives in the
 * controller — this hook owns what a component must get right around it:
 *
 * - one controller per mounted component, created once;
 * - `run` on every change of the inputs, with the cleanup cancelling only the timer — the next
 *   run (or the unmount path) owns the revealed flag;
 * - the flag lowered on unmount, so a reveal cannot outlive the component that raised it.
 */
export const useSelectedItemReveal = ({
  lastRenderedItemNameRef,
  $isTemporarilyShowingSelectedImage,
  marker,
  durationMs,
  mediaGraceMs,
  renderedItemName,
  isMediaReady,
  selectedItemName,
  shouldShowProgressInViewer,
  hasProgressImage,
  isProgressImageResolving,
}: UseSelectedItemRevealArgs): void => {
  const [revealController] = useState(() =>
    createSelectedItemRevealController({
      lastRenderedItemNameRef,
      marker,
      setRevealed: (revealed) => $isTemporarilyShowingSelectedImage.set(revealed),
      durationMs,
      mediaGraceMs,
    })
  );

  useEffect(() => {
    revealController.run({
      shouldShowProgressInViewer,
      hasProgressImage,
      isProgressImageResolving,
      renderedItemName,
      isMediaReady,
      selectedItemName,
    });
    return () => {
      revealController.clearTimer();
    };
  }, [
    hasProgressImage,
    isMediaReady,
    isProgressImageResolving,
    renderedItemName,
    revealController,
    selectedItemName,
    shouldShowProgressInViewer,
  ]);

  useEffect(() => {
    return () => {
      $isTemporarilyShowingSelectedImage.set(false);
    };
  }, [$isTemporarilyShowingSelectedImage]);
};

/**
 * Which item's media has actually painted, as readiness for the item being rendered.
 *
 * Deliberately not a boolean: a boolean would be reset from a different effect than the one that
 * reads it, and a passive effect's setState does not reach the next effect's closure in the same
 * commit — so a video -> video click would read the new name with the previous video's readiness
 * and lift the overlay onto a black element. Comparing names cannot go stale that way.
 */
export const usePaintedItemName = (itemName: string | null) => {
  const [paintedItemName, setPaintedItemName] = useState<string | null>(null);
  const onPainted = useCallback(() => {
    setPaintedItemName(itemName);
  }, [itemName]);
  return {
    isMediaReady: itemName !== null && paintedItemName === itemName,
    onPainted,
  };
};
