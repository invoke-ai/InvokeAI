import type { WidgetManifest } from '@workbench/widgetContracts';

import { BellIcon } from 'lucide-react';

export const notificationsWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['bottom'],
  bottomPanel: 'expandable',
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: BellIcon,
  id: 'notifications',
  label: (t) => t('widgets.labels.notifications'),
  load: () => import('./implementation').then((module) => module.widgetImplementation),
  version: 1,
};
