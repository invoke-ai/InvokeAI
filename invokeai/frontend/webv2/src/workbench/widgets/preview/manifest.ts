import type { WidgetManifest } from '@workbench/widgetContracts';

import { EyeIcon } from 'lucide-react';

import { previewSettingsContribution } from './settingsContribution';

export const previewWidgetManifest: WidgetManifest = {
  allowFloating: true,
  allowMultiple: false,
  allowedRegions: ['center', 'right'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: EyeIcon,
  id: 'preview',
  label: (t) => t('widgets.labels.preview'),
  load: () => import('./implementation').then((module) => module.widgetImplementation),
  settings: previewSettingsContribution,
  version: 1,
};
