import type { WidgetManifest } from '@workbench/widgetContracts';

import { MapIcon } from 'lucide-react';

export const imageMapWidgetManifest: WidgetManifest = {
  allowFloating: true,
  allowMultiple: false,
  allowedRegions: ['left', 'right', 'center'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: MapIcon,
  id: 'image-map',
  label: (t) => t('widgets.labels.imageMap'),
  load: () => import('./implementation').then((module) => module.widgetImplementation),
  loadHost: () => import('./ImageMapDataRuntime').then((module) => module.ImageMapDataRuntime),
  settingsSection: 'imageMap',
  version: 1,
};
