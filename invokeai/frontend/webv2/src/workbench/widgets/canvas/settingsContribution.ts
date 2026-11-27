import type { SettingDefinition, SettingsContribution } from '@platform/ui/settings/contracts';

import { CANVAS_SCROLL_INPUT_KEY, CANVAS_SCROLL_INPUTS, CANVAS_SETTINGS } from './canvasSettings';

export const canvasSettingsContribution: SettingsContribution = {
  id: 'canvas',
  label: (t) => t('widgets.labels.canvas'),
  fields: [
    {
      description: (t) => t('widgets.canvas.settings.scrollInputDescription'),
      group: (t) => t('widgets.canvas.settings.sections.behavior'),
      id: CANVAS_SCROLL_INPUT_KEY,
      keywords: 'trackpad touchpad mouse wheel pinch zoom pan',
      kind: 'select',
      label: (t) => t('widgets.canvas.settings.scrollInput'),
      options: CANVAS_SCROLL_INPUTS.map((value) => ({
        label: (t) => t(`widgets.canvas.settings.scrollInputOptions.${value}`),
        value,
      })),
      scope: 'project',
    },
    ...CANVAS_SETTINGS.map((setting): SettingDefinition => ({
      group: (t) => t(`widgets.canvas.settings.sections.${setting.section}`),
      id: setting.key,
      kind: 'boolean',
      label: (t) => t(setting.labelKey),
      scope: 'project',
      keywords: setting.key === 'showCheckerboard' ? 'transparency background' : undefined,
    })),
  ],
  quick: [CANVAS_SCROLL_INPUT_KEY, ...CANVAS_SETTINGS.map((setting) => setting.key)],
  load: () => import('./settingsBindings'),
};
