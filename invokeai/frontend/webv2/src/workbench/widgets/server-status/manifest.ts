import type { WidgetManifest } from '@workbench/widgetContracts';

import { PlugZapIcon } from 'lucide-react';

export const serverStatusWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['bottom'],
  bottomPanel: 'tooltip',
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: PlugZapIcon,
  id: 'server-status',
  label: (t) => t('widgets.labels.serverStatus'),
  load: () => import('./implementation').then((module) => module.widgetImplementation),
  version: 1,
};
