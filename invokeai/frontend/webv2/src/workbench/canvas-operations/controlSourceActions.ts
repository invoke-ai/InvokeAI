import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';

import type { ControlLayerIntentResult } from './operationTypes';

import { uploadCanvasImage } from './backend/canvasImages';
import { exportCanvasComposite, type CanvasCompositeExportEngine } from './exportCanvasComposite';
import { getCanvasOperations } from './operationAccess';

export type FillControlFromBboxResult =
  | { status: 'applied'; result: ControlLayerIntentResult; imageName: string }
  | { status: 'empty' | 'stale' | 'not-ready' | 'over-budget' };

/**
 * Uses the visible raster content inside the bbox as a control layer's image: the composite is uploaded as a control
 * image and replaces the layer's image at the bbox origin, which detects its map again. The upload stays with the
 * account that started it; a sign-out on the way reports `stale`.
 */
export const fillControlFromBbox = async (options: {
  engine: CanvasCompositeExportEngine & { readonly projectId: string };
  layerId: string;
  uploadImage?: typeof uploadCanvasImage;
}): Promise<FillControlFromBboxResult> => {
  const { engine, layerId, uploadImage = uploadCanvasImage } = options;
  const owner = captureAccountScope();
  try {
    const exported = await exportCanvasComposite(engine, 'bbox');
    assertAccountScopeCurrent(owner);
    if (exported.status !== 'ok') {
      return exported;
    }
    const uploaded = await uploadImage(exported.blob, {
      fileName: 'control.png',
      imageCategory: 'control',
      projectId: engine.projectId,
      signal: owner.signal,
    });
    assertAccountScopeCurrent(owner);
    const result = getCanvasOperations(engine).applyControlLayerIntent(layerId, {
      image: { height: uploaded.height, imageName: uploaded.imageName, width: uploaded.width },
      kind: 'replace-image',
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: exported.rect.x, y: exported.rect.y },
    });
    return { imageName: uploaded.imageName, result, status: 'applied' };
  } catch (error) {
    if (!isAccountScopeCurrent(owner)) {
      return { status: 'stale' };
    }
    throw error;
  }
};
