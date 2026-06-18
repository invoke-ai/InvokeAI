import type { WidgetManifest } from '@workbench/types';

import { ListOrderedIcon } from 'lucide-react';

import { QueueWidgetView } from './QueueWidgetView';

export const queueWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['right', 'bottom'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: ListOrderedIcon,
  id: 'queue',
  label: 'Queue',
  labelText: 'Queue',
  version: 1,
  view: QueueWidgetView,
};
