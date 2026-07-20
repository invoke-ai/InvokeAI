import type { WidgetManifest } from '@workbench/widgetContracts';

import { BugIcon } from 'lucide-react';

export const diagnosticsWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['bottom', 'right'],
  bottomPanel: 'expandable',
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: BugIcon,
  id: 'diagnostics',
  label: (t) => t('widgets.labels.diagnostics'),
  load: () => import('./implementation').then((module) => module.widgetImplementation),
  version: 1,
};
