import type { WidgetManifest } from '@workbench/widgetContracts';

import { WandSparklesIcon } from 'lucide-react';

import { canvasSettingsContribution } from './settingsContribution';

export const canvasWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['center'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  graphBearing: { defaultGraphId: 'canvas-graph', sourceId: 'canvas', surfaces: ['center'] },
  icon: WandSparklesIcon,
  id: 'canvas',
  label: (t) => t('widgets.labels.canvas'),
  load: () => import('./implementation').then((module) => module.widgetImplementation),
  settings: canvasSettingsContribution,
  version: 1,
};
