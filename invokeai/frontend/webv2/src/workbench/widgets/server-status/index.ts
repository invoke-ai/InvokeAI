import type { WidgetManifest } from '@workbench/types';

import { PlugZapIcon } from 'lucide-react';

import { ServerStatusWidgetView } from './ServerStatusWidgetView';

export const serverStatusWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['bottom'],
  bottomPanel: 'tooltip',
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: PlugZapIcon,
  id: 'server-status',
  label: 'Server Status',
  labelText: 'Server Status',
  version: 1,
  view: ServerStatusWidgetView,
};
