import type { WidgetManifest } from '@workbench/widgetContracts';

import { ImageUpscaleIcon } from 'lucide-react';

export const upscaleWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['left'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  graphBearing: { defaultGraphId: 'upscale-graph', sourceId: 'upscale', surfaces: ['left'] },
  icon: ImageUpscaleIcon,
  id: 'upscale',
  label: (t) => t('widgets.labels.upscale'),
  load: () => import('@features/upscale/widget').then((module) => module.widgetImplementation),
  version: 1,
};
