import type { GalleryImage, GalleryItem } from '@features/gallery';
import type { Project } from '@workbench/projectContracts';

import { getSelectedGalleryImageFromValues, getSelectedGalleryItemFromValues } from '@features/gallery/contracts';
import { getProjectWidgetValues } from '@workbench/widgetState';

export { getSelectedGalleryImageFromValues } from '@features/gallery/contracts';

export const getSelectedGalleryImage = (project: Project): GalleryImage | null =>
  getSelectedGalleryImageFromValues(getProjectWidgetValues(project, 'gallery'));

/** The selected image or video, for commands that apply to either. */
export const getSelectedGalleryItem = (project: Project): GalleryItem | null =>
  getSelectedGalleryItemFromValues(getProjectWidgetValues(project, 'gallery'));
