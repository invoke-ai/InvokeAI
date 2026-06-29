import type { WidgetManifest } from '@workbench/types';

import { ImageIcon } from 'lucide-react';

import { GalleryWidgetFooter } from './GalleryWidgetFooter';
import { GalleryWidgetView } from './GalleryWidgetView';

export const galleryWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['left', 'right', 'center', 'bottom'],
  bottomPanel: 'expandable',
  chrome: { header: 'hidden' },
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  footer: GalleryWidgetFooter,
  icon: ImageIcon,
  id: 'gallery',
  label: 'Gallery',
  labelText: 'Gallery',
  version: 1,
  view: GalleryWidgetView,
};
