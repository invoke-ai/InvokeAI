import type { Project } from '@workbench/projectContracts';
import type { CanvasEngineHandle } from '@workbench/widgets/canvas/useCanvasEngine';
import type { WorkbenchCanvasCommands, WorkbenchQueries } from '@workbench/workbenchStore';

import { galleryImages } from '@features/gallery';
import {
  importGalleryImagesToCanvas,
  type GalleryCanvasImportDestination,
  type ImportGalleryImagesResult,
} from '@workbench/canvas-operations/api';

import { orderCanvasImageDropImages } from './canvasImageDnd';

type CanvasImageDropEngine = Pick<CanvasEngineHandle, 'layers' | 'projectId'>;

export const executeCanvasImageDropImport = async ({
  destination,
  canvas,
  engine,
  getGalleryImages = galleryImages.resolveMany,
  queries,
  imageNames,
  importGalleryImages = importGalleryImagesToCanvas,
  project,
}: {
  destination: GalleryCanvasImportDestination;
  canvas: WorkbenchCanvasCommands;
  engine: CanvasImageDropEngine | null;
  getGalleryImages?: typeof galleryImages.resolveMany;
  queries: Pick<WorkbenchQueries, 'getProject' | 'isActiveProject'>;
  imageNames: readonly string[];
  importGalleryImages?: typeof importGalleryImagesToCanvas;
  project: Project;
}): Promise<ImportGalleryImagesResult> => {
  const fetchedImages = await getGalleryImages([...imageNames]);
  const images = orderCanvasImageDropImages(imageNames, fetchedImages);

  return importGalleryImages({
    applyCanvasMutation: canvas.apply,
    destination,
    engine,
    getProject: queries.getProject,
    images,
    isActiveProject: queries.isActiveProject,
    project,
  });
};
