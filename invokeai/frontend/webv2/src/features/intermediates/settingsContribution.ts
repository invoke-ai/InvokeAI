import type { SettingsContribution } from '@platform/ui/settings/contracts';

import { INTERMEDIATES_SETTING_ID } from './data/focus';

/** Metadata only: the manager and its transport load when the section opens. */
export const intermediatesSettingsContribution: SettingsContribution = {
  id: 'intermediates',
  label: (t) => t('intermediates.title'),
  fields: [
    {
      id: INTERMEDIATES_SETTING_ID,
      kind: 'custom',
      fill: true,
      label: (t) => t('intermediates.title'),
      description: (t) => t('intermediates.description'),
      keywords: 'intermediates cleanup disk space clear temporary delete',
      scope: 'none',
    },
  ],
  load: () =>
    import('./ui/IntermediatesSettingsField').then((module) => ({ Field: module.IntermediatesSettingsField })),
};
