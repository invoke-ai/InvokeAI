import type { WidgetManifest } from '@workbench/widgetContracts';

import { EyeIcon } from 'lucide-react';

export const previewWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['center', 'right'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: EyeIcon,
  id: 'preview',
  label: (t) => t('widgets.labels.preview'),
  load: () => import('./implementation').then((module) => module.widgetImplementation),
  version: 1,
};
