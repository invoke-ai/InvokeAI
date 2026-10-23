import type { GalleryImage } from '@features/gallery';
import type { CanvasExportCapability } from '@workbench/canvas-engine/api';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { Project } from '@workbench/projectContracts';

import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getDocumentLayer } from '@workbench/canvas-engine/api';
import { getControlImageState } from '@workbench/canvasLayerContent';

import type { ControlLayerIntentResult } from './operationTypes';

import { uploadCanvasImage } from './backend/canvasImages';
import { exportCanvasComposite, type CanvasCompositeExportEngine } from './exportCanvasComposite';
import {
  importGalleryImagesToCanvas,
  type GalleryImportEngine,
  type ImportGalleryImagesResult,
} from './importGalleryImages';
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

export type SendGalleryImageAsControlResult =
  | { status: 'filled'; layerId: string; layerName: string }
  | { status: 'created'; layerId: string }
  | Exclude<ImportGalleryImagesResult, { status: 'imported' }>;

type SendAsControlEngine = GalleryImportEngine & {
  readonly exports: Pick<CanvasExportCapability, 'hasExportableLayerContent'>;
};

/**
 * Uses a gallery image as a control layer's image. The selected control layer takes it when that layer has no image
 * yet and the project is the one on the live canvas (`filled`, detected like any replaced image); otherwise a new
 * control layer holds it (`created`), as the canvas import makes it. Every refusal of either path is reported, and a
 * `project` older than the latest document is `stale-document`.
 */
export const sendGalleryImageAsControl = async (options: {
  applyCanvasMutation: (projectId: string, mutation: CanvasProjectMutation) => boolean | void;
  engine: SendAsControlEngine | null;
  getProject: (projectId: string) => Project | null;
  image: GalleryImage;
  isActiveProject: (projectId: string) => boolean;
  project: Project;
}): Promise<SendGalleryImageAsControlResult> => {
  const { engine, getProject, image, isActiveProject, project } = options;
  const latest = getProject(project.id);
  if (!latest) {
    return { status: 'stale-project' };
  }
  if (latest.canvas.document !== project.canvas.document) {
    return { status: 'stale-document' };
  }

  const { document } = project.canvas;
  const selected = getDocumentLayer(document, document.selectedLayerId);
  const liveEngine = engine !== null && engine.projectId === project.id && isActiveProject(project.id) ? engine : null;
  if (
    liveEngine &&
    selected?.type === 'control' &&
    !selected.isLocked &&
    getControlImageState(selected, liveEngine.exports.hasExportableLayerContent(selected.id)) === 'empty'
  ) {
    const result = getCanvasOperations(liveEngine).applyControlLayerIntent(selected.id, {
      image: { height: image.height, imageName: image.imageName, width: image.width },
      kind: 'replace-image',
      // At the bbox origin, as every other canvas import places an image.
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: document.bbox.x, y: document.bbox.y },
    });
    switch (result) {
      case 'updated':
      case 'unchanged':
      case 'no-detection-input':
        return { layerId: selected.id, layerName: selected.name, status: 'filled' };
      case 'missing':
        return { status: 'stale-document' };
      case 'blocked':
      case 'over-budget':
        return { status: 'blocked' };
      case 'failed':
        throw new Error(`Could not use ${image.imageName} as the image of control layer ${selected.id}.`);
      case 'locked':
        // Locked by a group above it: the image goes to a new layer instead.
        break;
    }
  }

  const imported = await importGalleryImagesToCanvas({ ...options, destination: 'control', images: [image] });
  if (imported.status !== 'imported') {
    return imported;
  }
  const [layerId] = imported.layerIds;
  return layerId === undefined ? { status: 'empty' } : { layerId, status: 'created' };
};
