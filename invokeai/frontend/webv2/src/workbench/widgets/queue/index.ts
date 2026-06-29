import type { WidgetManifest } from '@workbench/types';

import { ListOrderedIcon } from 'lucide-react';

import { ModelCacheFooter } from './ModelCacheFooter';
import { QueueDataRuntime } from './QueueDataRuntime';
import { QueueHeaderActions } from './QueueHeaderActions';
import { QueueHeaderLabel } from './QueueHeaderLabel';
import { QueueHeaderMenu } from './QueueHeaderMenu';
import { QueueWidgetView } from './QueueWidgetView';

export const queueWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['right'],
  failurePolicy: {
    isolateRenderFailure: true,
    onRegistrationFailure: 'disable',
  },
  footer: ModelCacheFooter,
  headerActions: QueueHeaderActions,
  headerMenu: QueueHeaderMenu,
  host: QueueDataRuntime,
  icon: ListOrderedIcon,
  id: 'queue',
  label: QueueHeaderLabel,
  labelText: 'Queue',
  settingsSection: 'queue',
  version: 1,
  view: QueueWidgetView,
};
