/**
 * Create an image-sized canvas with raster layers at the origin. Before widgets mount, {@link
 * importGalleryImagesToCanvas} uses its reducer path.
 */

import type { GalleryImage } from '@features/gallery';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { Project } from '@workbench/projectContracts';

import { importGalleryImagesToCanvas, type ImportGalleryImagesResult } from './importGalleryImages';

export type CreateCanvasFromImagesResult = ImportGalleryImagesResult & { projectId: string | null };

export const createCanvasFromImages = async (options: {
  applyCanvasMutation: (projectId: string, mutation: CanvasProjectMutation) => boolean | void;
  createProject: () => Project;
  getProject: (projectId: string) => Project | null;
  isActiveProject: (projectId: string) => boolean;
  images: readonly GalleryImage[];
}): Promise<CreateCanvasFromImagesResult> => {
  const { applyCanvasMutation, createProject, getProject, images, isActiveProject } = options;
  if (images.length === 0) {
    return { projectId: null, status: 'empty' };
  }
  const created = createProject();
  const width = Math.max(...images.map((image) => image.width));
  const height = Math.max(...images.map((image) => image.height));
  applyCanvasMutation(created.id, { height, type: 'resizeCanvasDocument', width });
  applyCanvasMutation(created.id, { bbox: { height, width, x: 0, y: 0 }, type: 'setCanvasBbox' });

  const project = getProject(created.id);
  if (!project) {
    return { projectId: created.id, status: 'stale-project' };
  }
  const result = await importGalleryImagesToCanvas({
    applyCanvasMutation,
    destination: 'raster',
    engine: null,
    getProject,
    images,
    isActiveProject,
    project,
  });
  return { ...result, projectId: created.id };
};
