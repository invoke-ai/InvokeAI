import type { WidgetManifest } from '@workbench/widgetContracts';

import { SlidersHorizontalIcon } from 'lucide-react';

export const generateWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['left'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  graphBearing: { defaultGraphId: 'generate-graph', sourceId: 'generate', surfaces: ['left'] },
  icon: SlidersHorizontalIcon,
  id: 'generate',
  label: (t) => t('widgets.labels.generate'),
  load: () =>
    import('@features/generation/widget').then((module) => ({
      headerActions: module.GenerateHeaderActions,
      view: module.GenerateWidgetView,
    })),
  version: 1,
};
