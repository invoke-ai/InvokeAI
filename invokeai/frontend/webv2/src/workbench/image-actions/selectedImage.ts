import type { GalleryImage } from '@features/gallery';
import type { Project } from '@workbench/projectContracts';

import { getSelectedGalleryImageFromValues } from '@features/gallery/contracts';
import { getProjectWidgetValues } from '@workbench/widgetState';

export { getSelectedGalleryImageFromValues } from '@features/gallery/contracts';

export const getSelectedGalleryImage = (project: Project): GalleryImage | null =>
  getSelectedGalleryImageFromValues(getProjectWidgetValues(project, 'gallery'));
