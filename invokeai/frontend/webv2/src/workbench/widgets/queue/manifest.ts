import type { WidgetManifest } from '@workbench/widgetContracts';

import { ListOrderedIcon } from 'lucide-react';

export const queueWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['right'],
  failurePolicy: {
    isolateRenderFailure: true,
    onRegistrationFailure: 'disable',
  },
  hasHost: true,
  icon: ListOrderedIcon,
  id: 'queue',
  label: (t) => t('widgets.labels.queue'),
  load: () =>
    import('@features/queue/widget').then((module) => ({
      footer: module.ModelCacheFooter,
      headerActions: module.QueueHeaderActions,
      headerLabel: module.QueueHeaderLabel,
      headerMenu: module.QueueHeaderMenu,
      host: module.QueueDataRuntime,
      view: module.QueueWidgetView,
    })),
  settingsSection: 'queue',
  version: 1,
};
