import type { GalleryImage } from '@features/gallery';
import type { CanvasLayerCapability } from '@workbench/canvas-engine/api';
import type { CanvasImageUploadResult } from '@workbench/canvas-engine/document/imageUpload';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { Project } from '@workbench/projectContracts';

import { galleryImages } from '@features/gallery';

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
 * Composites the visible raster layers within the generation bbox, uploads
 * the PNG as a canvas-owned image, and either hands the upload back for the
 * global-reference flow or creates the destination layer at the bbox origin
 * via the shared gallery-import pipeline (which re-checks the document
 * captured in `project` against the latest state, covering the export/upload
 * window).
 */
export const createFromBbox = async (options: {
  applyCanvasMutation: (projectId: string, mutation: CanvasProjectMutation) => boolean | void;
  destination: CreateFromBboxDestination;
  engine: CreateFromBboxEngine;
  getProject: (projectId: string) => Project | null;
  isActiveProject: (projectId: string) => boolean;
  project: Project;
  resolveImages?: (imageNames: string[]) => Promise<GalleryImage[]>;
  uploadImage?: typeof uploadCanvasImage;
}): Promise<CreateFromBboxResult> => {
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

  const exported = await exportCanvasComposite(engine, 'bbox');
  if (exported.status !== 'ok') {
    return exported;
  }

  const uploaded = await uploadImage(exported.blob, {
    fileName: 'bbox.png',
    imageCategory: 'other',
    isIntermediate: false,
  });

  if (destination === 'global-reference') {
    return { image: uploaded, status: 'uploaded' };
  }

  const [galleryImage] = await resolveImages([uploaded.imageName]);
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

  if (imported.status === 'imported') {
    return { destination, layerIds: imported.layerIds, status: 'created' };
  }
  if (imported.status === 'empty') {
    // Unreachable: exactly one image is always passed. Kept for exhaustiveness.
    return { status: 'blocked' };
  }

  return imported;
};
