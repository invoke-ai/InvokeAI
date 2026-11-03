import type { WidgetManifest } from '@workbench/widgetContracts';

import { SlidersHorizontalIcon } from 'lucide-react';

export const generateWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['left'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  graphBearing: { defaultGraphId: 'generate-graph', sourceId: 'generate', surfaces: ['left'] },
  icon: SlidersHorizontalIcon,
  id: 'generate',
  label: (t) => t('widgets.labels.generate'),
  // The project's generation flags: they apply to every generation the project runs (Canvas and Upscale included),
  // and live in the project settings rather than this widget's values (see `PROJECT_SETTINGS_OWNER`).
  settings: {
    id: 'generate',
    label: (t) => t('widgets.labels.generate'),
    fields: (['useCpuNoise', 'useNsfwChecker', 'useWatermark'] as const).map((id) => ({
      id,
      label: (t) => t(`settings.catalog.${id}.label`),
      description: (t) => t(`settings.catalog.${id}.description`),
      keywords: id === 'useNsfwChecker' ? 'safety blur output' : id === 'useWatermark' ? 'output' : 'seed noise',
      kind: 'boolean',
      scope: 'project',
    })),
    quick: ['useCpuNoise', 'useNsfwChecker', 'useWatermark'],
    load: () => import('./settingsBindings'),
  },
  load: () =>
    import('@features/generation/widget').then((module) => ({
      headerActions: module.GenerateHeaderActions,
      view: module.GenerateWidgetView,
    })),
  version: 1,
};
