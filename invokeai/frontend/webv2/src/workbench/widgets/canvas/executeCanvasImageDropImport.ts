import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchCanvasCommands, WorkbenchQueries } from '@workbench/workbenchStore';

import { galleryImages } from '@features/gallery';
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
  destination,
  canvas,
  engine,
  getGalleryImages = galleryImages.resolveMany,
  queries,
  imageNames,
  importGalleryImages = importGalleryImagesToCanvas,
  project,
  sendAsControl = sendGalleryImageAsControl,
}: {
  destination: GalleryCanvasImportDestination;
  canvas: WorkbenchCanvasCommands;
  engine: CanvasImageDropEngine | null;
  getGalleryImages?: typeof galleryImages.resolveMany;
  queries: Pick<WorkbenchQueries, 'getProject' | 'isActiveProject'>;
  imageNames: readonly string[];
  importGalleryImages?: typeof importGalleryImagesToCanvas;
  project: Project;
  sendAsControl?: typeof sendGalleryImageAsControl;
}): Promise<CanvasImportNotice> => {
  const fetchedImages = await getGalleryImages([...imageNames]);
  const images = orderCanvasImageDropImages(imageNames, fetchedImages);
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
