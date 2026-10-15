import type { SettingsContribution } from '@platform/ui/settings/contracts';

export const imageMapSettingsContribution: SettingsContribution = {
  id: 'imageMap',
  label: (t) => t('widgets.labels.imageMap'),
  fields: [
    {
      id: 'showClusterLabels',
      kind: 'boolean',
      label: (t) => t('settingsDialog.fields.showClusterLabels'),
      scope: 'instance',
      keywords: 'tags text display',
    },
    {
      id: 'clusterEps',
      kind: 'custom',
      label: (t) => t('settingsDialog.fields.clusterStrength'),
      description: (t) => t('settingsDialog.fields.clusterStrengthHint'),
      scope: 'instance',
      keywords: 'dbscan eps clustering density granularity',
    },
    {
      id: 'clickSelectsCluster',
      kind: 'boolean',
      label: (t) => t('settingsDialog.fields.clickSelectsCluster'),
      scope: 'instance',
      keywords: 'click gallery selection',
    },
  ],
  load: () => import('./settingsBindings'),
};
