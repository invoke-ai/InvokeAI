import type { WidgetManifest } from '@workbench/widgetContracts';

import { loadQueueWidgetHost, loadQueueWidgetImplementation } from '@features/queue/widget';
import { ListOrderedIcon } from 'lucide-react';

export const queueWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['right'],
  failurePolicy: {
    isolateRenderFailure: true,
    onRegistrationFailure: 'disable',
  },
  icon: ListOrderedIcon,
  id: 'queue',
  label: (t) => t('widgets.labels.queue'),
  load: loadQueueWidgetImplementation,
  loadHost: loadQueueWidgetHost,
  settingsSection: 'queue',
  version: 1,
};
