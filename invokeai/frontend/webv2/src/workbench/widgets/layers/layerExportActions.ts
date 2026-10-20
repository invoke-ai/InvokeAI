import type { ExportBakedLayerBlobResult } from '@workbench/canvas-engine/api';

import { type ClipboardAccess, copyImage } from '@platform/browser/clipboard';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';

type LayerAssetExportFailureStatus = Exclude<ExportBakedLayerBlobResult['status'], 'ok'>;
type LayerAssetExportResult = { status: 'ok'; blob: Blob } | { status: LayerAssetExportFailureStatus };

interface CopyLayerToClipboardDeps extends ClipboardAccess {
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
  await copyImage(result.blob, clipboard);
  return 'ok';
};

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
