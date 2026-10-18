import type { GalleryNavigationSnapshot } from '@features/gallery/contracts';
import type { AccountScope } from '@platform/state/accountLifecycle';

import {
  claimGalleryNavigationSequence,
  getPersistedSelectedGalleryItemKeys,
  hasGalleryListingSearch,
  hasGalleryLocatorRequests,
  isGalleryNavigationCurrent,
  isGalleryNavigationStateUnchanged,
} from '@features/gallery/contracts';
import { isOpenQueueItem } from '@features/queue/contracts';
import { readQueueItemLastSavedImageName } from '@features/queue/queries';
import { getProjectWidgetValues } from '@workbench/widgetState';

import type { GalleryRevealContext } from './revealGalleryItem';

import { revealGalleryItem } from './revealGalleryItem';

export interface ExternalGenerationRevealContext extends GalleryRevealContext {
  /** Whether the user is typing: this window has focus and a text field in it does. */
  isEditingText: () => boolean;
  /** Whether this tab is on screen; a hidden tab defers its reveal until it is shown. */
  isVisible: () => boolean;
  /** Calls `listener` whenever the tab is shown or hidden; returns the unsubscribe. */
  onVisibilityChange: (listener: () => void) => () => void;
}

/** Captured when the completion arrives, before the reveal chunk loads. */
export interface ExternalGenerationRevealRequest {
  accountScope: AccountScope;
  /** The Gallery navigation at arrival; any change the user makes after it stands the reveal down. */
  navigation: GalleryNavigationSnapshot;
  /** The project on screen at arrival; the reveal never lands in another. */
  projectId: string;
  /** The backend queue item that completed. */
  queueItemId: number;
  /** Aborted when a newer completion supersedes this one or the composing runtime is disposed. */
  signal: AbortSignal;
}

export type ExternalGenerationRevealOutcome =
  /** Revealed, or stood down for good. */
  | { kind: 'settled' }
  /** Stood down only because the tab is hidden; resume from `sequence` once it is visible again. */
  | { kind: 'hidden'; sequence: number };

const SETTLED: ExternalGenerationRevealOutcome = { kind: 'settled' };

/**
 * Reveal another client's completed generation, as Find in Gallery would: select the board of its last saved image,
 * then the image, and announce it. It never takes the Gallery from the user. It stands down once a newer completion
 * arrives, while the user types in the focused window, while a run of their own in this project is open, while a
 * search or semantic query narrows the listing, while several items are selected, while another reveal is locating
 * its item, and once the user navigates the Gallery after arrival; all of these are dropped. A hidden tab only defers
 * it: the outcome asks to resume once the tab is visible. Every condition is checked again immediately before the
 * first Gallery change.
 *
 * `ownedSequence` is the latest navigation claim the reveal may treat as its own: the one at arrival, or on resuming
 * the one it claimed before the tab was hidden.
 */
export const revealExternalGeneration = async (
  context: ExternalGenerationRevealContext,
  request: ExternalGenerationRevealRequest,
  ownedSequence: number = request.navigation.sequence
): Promise<ExternalGenerationRevealOutcome> => {
  const { isEditingText, isVisible, queries } = context;
  const { accountScope, navigation, projectId, queueItemId, signal } = request;
  let owned = ownedSequence;
  // Every condition but visibility. A hidden tab cannot change the Gallery, so the arrival snapshot stays valid.
  const mayApplyWhenVisible = (): boolean => {
    if (signal.aborted || accountScope.signal.aborted || !queries.isActiveProject(projectId)) {
      return false;
    }

    const project = queries.getSnapshot().activeProject;
    const values = getProjectWidgetValues(project, 'gallery');

    return (
      !isEditingText() &&
      !project.queue.items.some(isOpenQueueItem) &&
      !hasGalleryListingSearch(values) &&
      getPersistedSelectedGalleryItemKeys(values).length <= 1 &&
      !hasGalleryLocatorRequests() &&
      isGalleryNavigationStateUnchanged(navigation, values)
    );
  };
  const mayApply = (): boolean => isVisible() && mayApplyWhenVisible();
  // Any navigation claimed since the one this reveal owns is the user's.
  const isOwned = (): boolean => isGalleryNavigationCurrent(owned);
  const standDown = (): ExternalGenerationRevealOutcome =>
    !isVisible() && isOwned() && mayApplyWhenVisible() ? { kind: 'hidden', sequence: owned } : SETTLED;

  if (!isOwned() || !mayApply()) {
    return standDown();
  }

  const imageName = await readQueueItemLastSavedImageName(queueItemId, signal);

  if (imageName === null) {
    return SETTLED;
  }
  if (!isOwned() || !mayApply()) {
    return standDown();
  }

  owned = claimGalleryNavigationSequence();
  await revealGalleryItem(
    context,
    { kind: 'image', name: imageName },
    { accountScope, announcement: 'external-generation', locatorSignal: signal, mayApply, projectId, sequence: owned }
  );

  // Revealing changes the Gallery's navigation, so only a reveal the tab's hiding stood down can resume.
  return standDown();
};
