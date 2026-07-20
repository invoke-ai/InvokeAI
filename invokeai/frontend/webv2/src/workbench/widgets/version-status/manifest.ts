import type { WidgetManifest } from '@workbench/widgetContracts';

import { InfoIcon } from 'lucide-react';

export const versionStatusWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['bottom'],
  bottomPanel: 'tooltip',
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: InfoIcon,
  id: 'version-status',
  label: (t) => t('widgets.labels.versionStatus'),
  load: () => import('./implementation').then((module) => module.widgetImplementation),
  version: 1,
};
