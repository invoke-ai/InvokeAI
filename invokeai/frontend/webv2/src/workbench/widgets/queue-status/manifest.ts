import type { WidgetManifest } from '@workbench/widgetContracts';

import { ListOrderedIcon } from 'lucide-react';

export const queueStatusWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['bottom'],
  bottomPanel: 'tooltip',
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: ListOrderedIcon,
  id: 'queue-status',
  label: (t) => t('widgets.labels.queueStatus'),
  load: () => import('./implementation').then((module) => module.widgetImplementation),
  version: 1,
};
