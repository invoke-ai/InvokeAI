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

interface SaveLayerToAssetsDeps {
  exportLayer(
    layerId: string,
    options: { includeDisabled: boolean; signal: AbortSignal }
  ): Promise<LayerAssetExportResult>;
  upload(file: File, boardId: string, options: { signal: AbortSignal }): Promise<unknown>;
}

export type SaveLayerToAssetsResult = 'saved' | 'stale' | LayerAssetExportFailureStatus;

/**
 * Owns the complete local-export → gallery-upload boundary. A stale account
 * reports `stale` instead of leaking an abort into the next account's UI.
 */
export const saveLayerToAssets = async (
  layerId: string,
  { exportLayer, upload }: SaveLayerToAssetsDeps
): Promise<SaveLayerToAssetsResult> => {
  const owner = captureAccountScope();

  try {
    const result = await exportLayer(layerId, { includeDisabled: true, signal: owner.signal });

    assertAccountScopeCurrent(owner);
    if (result.status !== 'ok') {
      return result.status;
    }

    const file = new File([result.blob], `layer-${layerId}.png`, { type: result.blob.type || 'image/png' });
    await upload(file, 'none', { signal: owner.signal });

    assertAccountScopeCurrent(owner);
    return 'saved';
  } catch (error) {
    if (!isAccountScopeCurrent(owner)) {
      return 'stale';
    }

    throw error;
  }
};
