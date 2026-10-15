import type { WidgetManifest } from '@workbench/widgetContracts';

import { gallerySettingsContribution } from '@features/gallery/settingsContribution';
import { ImageIcon } from 'lucide-react';

export const galleryWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['left', 'right', 'center', 'bottom'],
  bottomPanel: 'expandable',
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: ImageIcon,
  id: 'gallery',
  label: (t) => t('widgets.labels.gallery'),
  load: () =>
    import('@features/gallery/widget').then((module) => ({
      footer: module.GalleryWidgetFooter,
      headerLabel: module.GalleryWidgetLabel,
      view: module.GalleryWidgetView,
    })),
  settings: gallerySettingsContribution,
  version: 1,
};
