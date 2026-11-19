import type { WidgetManifest } from '@workbench/widgetContracts';

import { serverOutputSettings } from '@workbench/settings/applicationContributions';
import { SquareTerminalIcon } from 'lucide-react';

export const serverOutputWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['bottom'],
  bottomPanel: 'expandable',
  chrome: { body: 'fill' },
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: SquareTerminalIcon,
  id: 'server-output',
  label: (t) => t('widgets.labels.serverOutput'),
  load: () => import('./implementation').then((module) => module.widgetImplementation),
  requiresCapability: 'viewServerOutput',
  settings: serverOutputSettings,
  version: 1,
};
