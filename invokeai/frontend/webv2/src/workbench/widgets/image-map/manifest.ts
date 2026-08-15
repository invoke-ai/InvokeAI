import type { WidgetManifest } from '@workbench/widgetContracts';

import { MapIcon } from 'lucide-react';

export const imageMapWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['left', 'right', 'center'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: MapIcon,
  id: 'image-map',
  label: (t) => t('widgets.labels.imageMap'),
  load: () => import('./implementation').then((module) => module.widgetImplementation),
  version: 1,
};
