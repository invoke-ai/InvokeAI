import type { ExportBakedLayerBlobResult } from '@workbench/canvas-engine/api';

import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';

type ClipboardLike = { write(items: ClipboardItem[]): Promise<void> };
type ClipboardItemCtor = new (items: Record<string, Blob>) => ClipboardItem;

export interface CopyBlobToClipboardDeps {
  clipboard?: ClipboardLike;
  ClipboardItemCtor?: ClipboardItemCtor;
}

export const copyBlobToClipboard = async (blob: Blob, deps: CopyBlobToClipboardDeps = {}): Promise<void> => {
  const clipboard = deps.clipboard ?? globalThis.navigator?.clipboard;
  const ClipboardItemImpl = deps.ClipboardItemCtor ?? globalThis.ClipboardItem;
  if (!clipboard || !ClipboardItemImpl) {
    throw new Error('Clipboard image copy is unavailable');
  }
  const type = blob.type || 'image/png';
  await clipboard.write([new ClipboardItemImpl({ [type]: blob })]);
};

type LayerAssetExportFailureStatus = Exclude<ExportBakedLayerBlobResult['status'], 'ok'>;
type LayerAssetExportResult = { status: 'ok'; blob: Blob } | { status: LayerAssetExportFailureStatus };

interface CopyLayerToClipboardDeps extends CopyBlobToClipboardDeps {
  exportLayer(layerId: string, options: { includeDisabled: boolean }): Promise<LayerAssetExportResult>;
}

/** Bakes the layer (disabled or not) to PNG and puts it on the clipboard; a failed export reports its status. */
export const copyLayerToClipboard = async (
  layerId: string,
  { exportLayer, ...clipboard }: CopyLayerToClipboardDeps
): Promise<'ok' | LayerAssetExportFailureStatus> => {
  const result = await exportLayer(layerId, { includeDisabled: true });
  if (result.status !== 'ok') {
    return result.status;
  }
  await copyBlobToClipboard(result.blob, clipboard);
  return 'ok';
};

interface SaveLayerToBoardDeps {
  exportLayer(
    layerId: string,
    options: { includeDisabled: boolean; signal: AbortSignal }
  ): Promise<LayerAssetExportResult>;
  upload(file: File, boardId: string, options: { signal: AbortSignal }): Promise<unknown>;
}

export type SaveLayerToBoardResult = 'saved' | 'stale' | LayerAssetExportFailureStatus;

/**
 * Owns the complete local-export → gallery-upload boundary: the baked layer is uploaded to `boardId` (`'none'` for
 * Uncategorized). A stale account reports `stale` instead of leaking an abort into the next account's UI.
 */
export const saveLayerToBoard = async (
  layerId: string,
  boardId: string,
  { exportLayer, upload }: SaveLayerToBoardDeps
): Promise<SaveLayerToBoardResult> => {
  const owner = captureAccountScope();

  try {
    const result = await exportLayer(layerId, { includeDisabled: true, signal: owner.signal });

    assertAccountScopeCurrent(owner);
    if (result.status !== 'ok') {
      return result.status;
    }

    const file = new File([result.blob], `layer-${layerId}.png`, { type: result.blob.type || 'image/png' });
    await upload(file, boardId, { signal: owner.signal });

    assertAccountScopeCurrent(owner);
    return 'saved';
  } catch (error) {
    if (!isAccountScopeCurrent(owner)) {
      return 'stale';
    }

    throw error;
  }
};
