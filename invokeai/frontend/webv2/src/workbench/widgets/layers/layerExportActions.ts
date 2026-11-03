import type { QueryClient } from '@tanstack/react-query';
import type { AdjustmentRefusal, ExportBakedLayerBlobResult, LayerUnreadable } from '@workbench/canvas-engine/api';

import { canOnBoard } from '@features/gallery/contracts';
import {
  canAddToCachedGalleryBoard,
  ensureGalleryBoardDestinations,
  getCachedGalleryBoard,
  invalidateGallery,
  invalidateGalleryBoardDestinations,
  invalidateGalleryItems,
} from '@features/gallery/queries';
import { type ClipboardAccess, copyImage } from '@platform/browser/clipboard';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';

type LayerAssetExportFailureStatus = Exclude<
  ExportBakedLayerBlobResult['status'],
  'ok' | 'adjustment-refused' | 'unreadable'
>;
type LayerAssetExportResult =
  | { status: 'ok'; blob: Blob }
  | { status: LayerAssetExportFailureStatus }
  | LayerUnreadable
  | AdjustmentRefusal;

interface CopyLayerToClipboardDeps extends ClipboardAccess {
  exportLayer(layerId: string, options: { includeDisabled: boolean }): Promise<LayerAssetExportResult>;
}

/** Bakes the layer (disabled or not) to PNG and puts it on the clipboard; a failed export reports its status. */
export const copyLayerToClipboard = async (
  layerId: string,
  { exportLayer, ...clipboard }: CopyLayerToClipboardDeps
): Promise<'ok' | LayerAssetExportFailureStatus | LayerUnreadable | AdjustmentRefusal> => {
  const result = await exportLayer(layerId, { includeDisabled: true });
  if (result.status === 'adjustment-refused' || result.status === 'unreadable') {
    return result;
  }
  if (result.status !== 'ok') {
    return result.status;
  }
  await copyImage(result.blob, clipboard);
  return 'ok';
};

interface SaveLayerToBoardDeps {
  client: QueryClient;
  exportLayer(
    layerId: string,
    options: { includeDisabled: boolean; signal: AbortSignal }
  ): Promise<LayerAssetExportResult>;
  upload(file: File, boardId: string, options: { signal: AbortSignal }): Promise<unknown>;
}

export type SaveLayerToBoardResult =
  | 'saved'
  | 'stale'
  | 'board-view-only'
  | 'board-unavailable'
  | LayerAssetExportFailureStatus
  | LayerUnreadable
  | AdjustmentRefusal;

/**
 * Owns the complete local-export → gallery-upload boundary: the baked layer is uploaded to `boardId` (`'none'` for
 * Uncategorized). A real board is judged first from the ensured listing of every readable board, as Gallery uploads
 * are: one listed without `add_media` reports `board-view-only` and one no longer listed `board-unavailable`, before
 * anything is baked, and the listings are read again so the picker stops offering it. Once the upload settles the
 * gallery reads again; a failed one also re-reads the listings (the board may have refused it) and rethrows. A stale
 * account reports `stale` instead of leaking an abort into the next account's UI.
 */
export const saveLayerToBoard = async (
  layerId: string,
  boardId: string,
  { client, exportLayer, upload }: SaveLayerToBoardDeps
): Promise<SaveLayerToBoardResult> => {
  const owner = captureAccountScope();
  const isRealBoard = boardId !== 'none';

  if (isRealBoard) {
    await ensureGalleryBoardDestinations(client);
    if (!isAccountScopeCurrent(owner)) {
      return 'stale';
    }
    if (!canAddToCachedGalleryBoard(boardId, client)) {
      const cachedBoard = getCachedGalleryBoard(boardId, client);
      void invalidateGalleryBoardDestinations(client, owner);
      return cachedBoard && !canOnBoard(cachedBoard, 'add_media') ? 'board-view-only' : 'board-unavailable';
    }
  }

  try {
    const result = await exportLayer(layerId, { includeDisabled: true, signal: owner.signal });

    assertAccountScopeCurrent(owner);
    if (result.status === 'adjustment-refused' || result.status === 'unreadable') {
      return result;
    }
    if (result.status !== 'ok') {
      return result.status;
    }

    const file = new File([result.blob], `layer-${layerId}.png`, { type: result.blob.type || 'image/png' });
    try {
      await upload(file, boardId, { signal: owner.signal });
    } catch (error) {
      if (isAccountScopeCurrent(owner)) {
        // A board that refused the upload re-reads the listings (destination read included) along with the items.
        if (isRealBoard) {
          void invalidateGalleryItems(client, owner);
          void invalidateGalleryBoardDestinations(client, owner);
        } else {
          void invalidateGallery(client, owner);
        }
      }
      throw error;
    }

    assertAccountScopeCurrent(owner);
    void invalidateGallery(client, owner);
    return 'saved';
  } catch (error) {
    if (!isAccountScopeCurrent(owner)) {
      return 'stale';
    }

    throw error;
  }
};
