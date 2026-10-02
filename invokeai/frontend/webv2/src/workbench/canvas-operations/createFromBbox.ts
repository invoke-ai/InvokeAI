import type { GalleryImage } from '@features/gallery';
import type { CanvasLayerCapability } from '@workbench/canvas-engine/api';
import type { CanvasImageUploadResult } from '@workbench/canvas-engine/document/imageUpload';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { Project } from '@workbench/projectContracts';

import { galleryImages } from '@features/gallery';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';

import type { CanvasCompositeExportEngine } from './exportCanvasComposite';
import type { GalleryCanvasImportDestination } from './importGalleryImages';

import { uploadCanvasImage } from './backend/canvasImages';
import { exportCanvasComposite } from './exportCanvasComposite';
import { importGalleryImagesToCanvas } from './importGalleryImages';

export type CreateFromBboxLayerDestination = Extract<
  GalleryCanvasImportDestination,
  'raster' | 'control' | 'regional-reference'
>;

export type CreateFromBboxDestination = 'global-reference' | CreateFromBboxLayerDestination;

export type CreateFromBboxResult =
  | { status: 'created'; destination: CreateFromBboxLayerDestination; layerIds: string[] }
  | { status: 'uploaded'; image: CanvasImageUploadResult }
  | { status: 'empty' | 'stale' | 'not-ready' | 'over-budget' }
  | { status: 'blocked' | 'stale-document' | 'stale-project' };

type CreateFromBboxEngine = CanvasCompositeExportEngine & {
  readonly layers: CanvasLayerCapability;
  readonly projectId: string;
};

/**
 * Upload visible raster content inside the bbox for a global reference or layer at the bbox origin. The shared
 * import pipeline rechecks document identity after export/upload.
 */
export const createFromBbox = async (options: {
  applyCanvasMutation: (projectId: string, mutation: CanvasProjectMutation) => boolean | void;
  destination: CreateFromBboxDestination;
  engine: CreateFromBboxEngine;
  getProject: (projectId: string) => Project | null;
  isActiveProject: (projectId: string) => boolean;
  project: Project;
  resolveImages?: (imageNames: string[], signal?: AbortSignal) => Promise<GalleryImage[]>;
  uploadImage?: typeof uploadCanvasImage;
}): Promise<CreateFromBboxResult> => {
  const owner = captureAccountScope();
  const {
    applyCanvasMutation,
    destination,
    engine,
    getProject,
    isActiveProject,
    project,
    resolveImages = galleryImages.resolveMany,
    uploadImage = uploadCanvasImage,
  } = options;

  try {
    const exported = await exportCanvasComposite(engine, 'bbox');

    assertAccountScopeCurrent(owner);
    if (exported.status !== 'ok') {
      return exported;
    }

    const uploaded = await uploadImage(exported.blob, {
      fileName: 'bbox.png',
      imageCategory: 'other',
      isIntermediate: false,
      signal: owner.signal,
    });

    assertAccountScopeCurrent(owner);
    if (destination === 'global-reference') {
      return { image: uploaded, status: 'uploaded' };
    }

    const [galleryImage] = await resolveImages([uploaded.imageName], owner.signal);

    assertAccountScopeCurrent(owner);
    if (!galleryImage) {
      throw new Error(`Uploaded bbox composite ${uploaded.imageName} could not be resolved to a gallery image`);
    }

    const imported = await importGalleryImagesToCanvas({
      applyCanvasMutation,
      destination,
      engine,
      getProject,
      images: [galleryImage],
      isActiveProject,
      project,
    });

    assertAccountScopeCurrent(owner);
    if (imported.status === 'imported') {
      return { destination, layerIds: imported.layerIds, status: 'created' };
    }
    switch (imported.status) {
      case 'blocked':
      case 'stale-document':
      case 'stale-project':
        return { status: imported.status };
      case 'empty':
        // Unreachable: exactly one image is always passed. Kept for exhaustiveness.
        return { status: 'blocked' };
      case 'capabilities-unavailable':
        // Unreachable: only 'control-resized' consults the capability table, and this flow never
        // asks for it -- CreateFromBboxLayerDestination excludes that destination.
        return { status: 'blocked' };
    }
  } catch (error) {
    if (!isAccountScopeCurrent(owner)) {
      return { status: 'stale' };
    }

    throw error;
  }
};
