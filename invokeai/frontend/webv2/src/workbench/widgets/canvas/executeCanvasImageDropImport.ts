import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchCanvasCommands, WorkbenchQueries } from '@workbench/workbenchStore';

import { galleryImages, type GalleryImage } from '@features/gallery';
import {
  getCanvasImportNotice,
  getSendAsControlNotice,
  importGalleryImagesToCanvas,
  sendGalleryImageAsControl,
  type CanvasImportNotice,
  type GalleryCanvasImportDestination,
} from '@workbench/canvas-operations/api';

import { orderCanvasImageDropImages } from './canvasImageDnd';

type CanvasImageDropEngine = Pick<CanvasEngineHandle, 'document' | 'exports' | 'layers' | 'projectId'>;

/** Imports the dropped gallery images and says how it went, as the gallery menu's matching action does. */
export const executeCanvasImageDropImport = async ({
  getGalleryImages = galleryImages.resolveMany,
  imageNames,
  ...options
}: Omit<CanvasDropImportOptions, 'images'> & {
  getGalleryImages?: typeof galleryImages.resolveMany;
  imageNames: readonly string[];
}): Promise<CanvasImportNotice> => {
  const fetchedImages = await getGalleryImages([...imageNames]);
  return importCanvasDropImages({ ...options, images: orderCanvasImageDropImages(imageNames, fetchedImages) });
};

interface CanvasDropImportOptions {
  destination: GalleryCanvasImportDestination;
  canvas: WorkbenchCanvasCommands;
  engine: CanvasImageDropEngine | null;
  queries: Pick<WorkbenchQueries, 'getProject' | 'isActiveProject'>;
  images: readonly GalleryImage[];
  importGalleryImages?: typeof importGalleryImagesToCanvas;
  project: Project;
  sendAsControl?: typeof sendGalleryImageAsControl;
}

/** Lands images dropped on a canvas zone, gallery images or files just uploaded, and says how it went. */
export const importCanvasDropImages = async ({
  destination,
  canvas,
  engine,
  queries,
  images,
  importGalleryImages = importGalleryImagesToCanvas,
  project,
  sendAsControl = sendGalleryImageAsControl,
}: CanvasDropImportOptions): Promise<CanvasImportNotice> => {
  const shared = {
    applyCanvasMutation: canvas.apply,
    engine,
    getProject: queries.getProject,
    isActiveProject: queries.isActiveProject,
    project,
  };

  // One image dropped as a control layer fills the selected control layer still waiting for one, as Use as Control
  // does; the resized zone always adds a layer, since its image is made for the generation size.
  const [image] = images;
  if (destination === 'control' && images.length === 1 && image) {
    return getSendAsControlNotice(await sendAsControl({ ...shared, image }));
  }
  return getCanvasImportNotice(await importGalleryImages({ ...shared, destination, images }));
};
