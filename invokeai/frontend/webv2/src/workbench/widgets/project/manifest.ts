import type { WidgetManifest } from '@workbench/widgetContracts';

import { FolderCogIcon } from 'lucide-react';

export const projectWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['left', 'right'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: FolderCogIcon,
  id: 'project',
  label: (t) => t('widgets.labels.project'),
  load: () => import('./implementation').then((module) => module.widgetImplementation),
  version: 1,
};
