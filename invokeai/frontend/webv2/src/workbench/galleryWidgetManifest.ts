import type { WidgetManifest } from '@workbench/widgetContracts';

import { ImageIcon } from 'lucide-react';

export const galleryWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['left', 'right', 'center', 'bottom'],
  bottomPanel: 'expandable',
  chrome: { header: 'hidden' },
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: ImageIcon,
  id: 'gallery',
  label: (t) => t('widgets.labels.gallery'),
  load: () =>
    import('@features/gallery/widget').then((module) => ({
      footer: module.GalleryWidgetFooter,
      view: module.GalleryWidgetView,
    })),
  version: 1,
};
