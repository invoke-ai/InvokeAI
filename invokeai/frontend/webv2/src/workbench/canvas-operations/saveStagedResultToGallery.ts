import type { GalleryImage } from '@features/gallery';
import type { Project } from '@workbench/projectContracts';

import { galleryDurability, galleryOrganization } from '@features/gallery';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';

import { getCanvasSaveBoardId } from './saveCanvasToGallery';

/** `boardId` is where the image is meant to be; `'none'` is Uncategorized. */
export type SaveStagedResultOutcome =
  | { status: 'saved'; imageName: string; boardId: string }
  /** In the gallery (and so findable), but not on the board it was meant for. */
  | { status: 'board-failed'; imageName: string; boardId: string }
  | { status: 'stale' };

interface StagedResultTransports {
  addToBoard?: (boardId: string, imageNames: string[], signal?: AbortSignal) => Promise<string[]>;
  promote?: (imageName: string) => Promise<GalleryImage>;
}

/** Puts a promoted image on `boardId` unless it is there already; the backend upserts, so a repeat is harmless. */
const assignBoard = async (
  { addToBoard = galleryOrganization.addToBoard }: StagedResultTransports,
  imageName: string,
  boardId: string,
  currentBoardId: string
): Promise<SaveStagedResultOutcome> => {
  if (boardId === currentBoardId) {
    return { boardId, imageName, status: 'saved' };
  }

  const owner = captureAccountScope();

  try {
    const added = await addToBoard(boardId, [imageName], owner.signal);

    assertAccountScopeCurrent(owner);
    return added.includes(imageName)
      ? { boardId, imageName, status: 'saved' }
      : { boardId, imageName, status: 'board-failed' };
  } catch {
    return isAccountScopeCurrent(owner) ? { boardId, imageName, status: 'board-failed' } : { status: 'stale' };
  }
};

/**
 * Promote a staged canvas result into the gallery with the same destination as a canvas save: a board the image is
 * already on stays; otherwise the project's auto-add board, else Uncategorized. The destination is read from the
 * project before anything is awaited, so switching projects mid-save cannot redirect it. A failed promotion throws;
 * a failed board assignment is reported as such, since the image is in the gallery either way.
 */
export const saveStagedResultToGallery = async (
  { imageName, project }: { imageName: string; project: Project },
  transports: StagedResultTransports = {}
): Promise<SaveStagedResultOutcome> => {
  const owner = captureAccountScope();
  const canvasBoardId = getCanvasSaveBoardId(project) ?? 'none';
  const { promote = galleryDurability.save } = transports;
  let record: GalleryImage;

  try {
    record = await promote(imageName);
    assertAccountScopeCurrent(owner);
  } catch (error) {
    if (!isAccountScopeCurrent(owner)) {
      return { status: 'stale' };
    }

    throw error;
  }

  const currentBoardId = record.boardId ?? 'none';

  return assignBoard(transports, imageName, currentBoardId !== 'none' ? currentBoardId : canvasBoardId, currentBoardId);
};

/** Try the board assignment of a save that reported `board-failed` again. */
export const retryStagedResultBoard = (
  { boardId, imageName }: { boardId: string; imageName: string },
  transports: StagedResultTransports = {}
): Promise<SaveStagedResultOutcome> => assignBoard(transports, imageName, boardId, 'none');
