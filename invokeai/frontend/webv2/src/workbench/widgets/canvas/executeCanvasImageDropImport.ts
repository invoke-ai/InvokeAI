import type { Project, WorkbenchState } from '@workbench/types';
import type { CanvasEngineHandle } from '@workbench/widgets/canvas/useCanvasEngine';
import type { WorkbenchAction } from '@workbench/workbenchState';
import type { Dispatch } from 'react';

import {
  importGalleryImagesToCanvas,
  type GalleryCanvasImportDestination,
  type ImportGalleryImagesResult,
} from '@workbench/canvas-operations/api';
import { getGalleryImagesByNames } from '@workbench/gallery/api';

import { orderCanvasImageDropImages } from './canvasImageDnd';

type CanvasImageDropEngine = Pick<CanvasEngineHandle, 'layers' | 'projectId'>;

export const executeCanvasImageDropImport = async ({
  destination,
  dispatch,
  engine,
  getGalleryImages = getGalleryImagesByNames,
  getState,
  imageNames,
  importGalleryImages = importGalleryImagesToCanvas,
  project,
}: {
  destination: GalleryCanvasImportDestination;
  dispatch: Dispatch<WorkbenchAction>;
  engine: CanvasImageDropEngine | null;
  getGalleryImages?: typeof getGalleryImagesByNames;
  getState: () => WorkbenchState;
  imageNames: readonly string[];
  importGalleryImages?: typeof importGalleryImagesToCanvas;
  project: Project;
}): Promise<ImportGalleryImagesResult> => {
  const fetchedImages = await getGalleryImages([...imageNames]);
  const images = orderCanvasImageDropImages(imageNames, fetchedImages);

  return importGalleryImages({ destination, dispatch, engine, getState, images, project });
};
