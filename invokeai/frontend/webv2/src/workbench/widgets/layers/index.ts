import type { WidgetManifest } from '@workbench/widgetContracts';

import { LayersIcon } from 'lucide-react';

export const layersWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['right'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: LayersIcon,
  id: 'layers',
  label: (t) => t('widgets.labels.layers'),
  load: () => import('./implementation').then((module) => module.widgetImplementation),
  version: 1,
};
