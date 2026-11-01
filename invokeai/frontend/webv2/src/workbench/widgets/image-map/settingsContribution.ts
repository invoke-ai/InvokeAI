import type { SettingsContribution } from '@platform/ui/settings/contracts';

export const imageMapSettingsContribution: SettingsContribution = {
  id: 'imageMap',
  label: (t) => t('widgets.labels.imageMap'),
  fields: [
    {
      id: 'showClusterLabels',
      kind: 'boolean',
      label: (t) => t('settingsDialog.fields.showClusterLabels'),
      scope: 'project',
      keywords: 'tags text display',
    },
    {
      id: 'clusterEps',
      kind: 'custom',
      label: (t) => t('settingsDialog.fields.clusterStrength'),
      description: (t) => t('settingsDialog.fields.clusterStrengthHint'),
      scope: 'project',
      keywords: 'dbscan eps clustering density granularity',
    },
    {
      id: 'clickSelectsCluster',
      kind: 'boolean',
      label: (t) => t('settingsDialog.fields.clickSelectsCluster'),
      scope: 'project',
      keywords: 'click gallery selection',
    },
  ],
  load: () => import('./settingsBindings'),
};
