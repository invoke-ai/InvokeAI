import type { WidgetManifest } from '@workbench/types';

import { ImageUpscaleIcon } from 'lucide-react';

import { UpscaleWidgetView } from './UpscaleWidgetView';

export const upscaleWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['left'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  graphBearing: { defaultGraphId: 'upscale-graph', sourceId: 'upscale', surfaces: ['left'] },
  icon: ImageUpscaleIcon,
  id: 'upscale',
  label: (t) => t('widgets.labels.upscale'),
  version: 1,
  view: UpscaleWidgetView,
};
