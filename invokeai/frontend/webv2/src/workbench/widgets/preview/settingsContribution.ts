import type { SettingsContribution } from '@platform/ui/settings/contracts';

export const previewSettingsContribution: SettingsContribution = {
  id: 'preview',
  label: (t) => t('widgets.labels.preview'),
  fields: [
    {
      id: 'filmstripVisible',
      kind: 'boolean',
      label: (t) => t('widgets.preview.showFilmstrip'),
      scope: 'instance',
      keywords: 'thumbnails gallery',
    },
    {
      id: 'comparisonMode',
      kind: 'select',
      label: (t) => t('settingsDialog.fields.comparisonMode'),
      scope: 'instance',
      options: [
        { label: (t) => t('widgets.preview.slider'), value: 'slider' },
        { label: (t) => t('widgets.preview.sideBySide'), value: 'side-by-side' },
        { label: (t) => t('widgets.preview.hover'), value: 'hover' },
      ],
    },
  ],
  // Both fit the gear's popover; the full dialog stays one click further.
  quick: ['filmstripVisible', 'comparisonMode'],
  load: () => import('./settingsBindings'),
};
