import { useStore } from '@nanostores/react';
import { $gallerySelection } from 'features/gallery/store/gallerySelectionSource';
import { useCallback, useEffect, useState } from 'react';

import type { SelectedItemRevealMachine } from './selectedItemReveal';

type UseSelectedItemRevealArgs = {
  /** The viewer context's machine. One instance serves both preview components, so a click that
   * switches media type is just another selection rather than a component swap to reason about. */
  revealMachine: SelectedItemRevealMachine;
  /** The item the component is rendering right now, or null when it has nothing to show. */
  renderedItemName: string | null;
  /** Whether that item has a frame on screen. See usePaintedItemName. */
  isMediaReady: boolean;
  shouldShowProgressInViewer: boolean;
  hasProgressImage: boolean;
  isProgressImageResolving: boolean;
};

/**
 * The preview components' reveal wiring, extracted so it can be mounted and tested with real
 * effect lifecycles (see useSelectedItemReveal.test.tsx). The sequencing lives in the machine —
 * this hook owns what a component must get right around it:
 *
 * - registering as a live driver while mounted (attach), so selections that land with neither
 *   preview mounted are settled by the provider instead of replaying as a reveal on return;
 * - syncing the machine on every change of the inputs, with the current selection descriptor.
 *
 * Deliberately nothing else: the machine owns the revealed flag and its timers exclusively —
 * components lowering the flag on unmount is how a provider-owned machine gets desynced.
 */
export const useSelectedItemReveal = ({
  revealMachine,
  renderedItemName,
  isMediaReady,
  shouldShowProgressInViewer,
  hasProgressImage,
  isProgressImageResolving,
}: UseSelectedItemRevealArgs): void => {
  const selection = useStore($gallerySelection);

  useEffect(() => revealMachine.attach(), [revealMachine]);

  useEffect(() => {
    revealMachine.sync({
      selection,
      renderedItemName,
      isMediaReady,
      shouldShowProgressInViewer,
      hasProgressImage,
      isProgressImageResolving,
    });
  }, [
    hasProgressImage,
    isMediaReady,
    isProgressImageResolving,
    renderedItemName,
    revealMachine,
    selection,
    shouldShowProgressInViewer,
  ]);
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
