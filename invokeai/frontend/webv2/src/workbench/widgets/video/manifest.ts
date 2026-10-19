import type { WidgetManifest } from '@workbench/widgetContracts';

import { ClapperboardIcon } from 'lucide-react';

export const videoWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['left'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  graphBearing: { defaultGraphId: 'video-graph', sourceId: 'video', surfaces: ['left'] },
  icon: ClapperboardIcon,
  id: 'video',
  label: (t) => t('widgets.labels.video'),
  load: () => import('@features/video/widget').then((module) => module.widgetImplementation),
  version: 1,
};
