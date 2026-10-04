import type { SettingsContribution } from '@platform/ui/settings/contracts';

import { CANVAS_SETTINGS } from './canvasSettings';

export const canvasSettingsContribution: SettingsContribution = {
  id: 'canvas',
  label: (t) => t('widgets.labels.canvas'),
  fields: CANVAS_SETTINGS.map((setting) => ({
    group: (t) => t(`widgets.canvas.settings.sections.${setting.section}`),
    id: setting.key,
    kind: 'boolean',
    label: (t) => t(setting.labelKey),
    scope: 'instance',
    keywords: setting.key === 'showCheckerboard' ? 'transparency background' : undefined,
  })),
  quick: CANVAS_SETTINGS.map((setting) => setting.key),
  load: () => import('./settingsBindings'),
};
