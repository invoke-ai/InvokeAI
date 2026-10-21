import type { WidgetManifest } from '@workbench/widgetContracts';

import { NetworkIcon } from 'lucide-react';

export const remoteWorkersWidgetManifest: WidgetManifest = {
  allowFloating: true,
  allowMultiple: false,
  allowedRegions: ['left', 'right'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: NetworkIcon,
  id: 'remote-workers',
  label: (t) => t('widgets.labels.remoteWorkers'),
  load: () => import('./implementation').then((module) => module.widgetImplementation),
  version: 1,
};
