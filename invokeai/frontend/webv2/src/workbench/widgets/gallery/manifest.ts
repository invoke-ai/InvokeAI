import type { WidgetManifest } from '@workbench/widgetContracts';

import { galleryQuickSettings, gallerySettingsFields } from '@features/gallery/settingsContribution';
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
      headerActions: module.GalleryWidgetHeaderActions,
      headerLabel: module.GalleryWidgetLabel,
      view: module.GalleryWidgetView,
    })),
  // The Gallery owns the field metadata; the binding is Workbench's (see `workbench/settings/README.md`).
  settings: {
    id: 'gallery',
    label: (t) => t('widgets.labels.gallery'),
    fields: gallerySettingsFields,
    quick: galleryQuickSettings,
    load: () => import('./settingsBindings'),
  },
  version: 1,
};
