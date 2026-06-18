import type { WidgetManifest } from '@workbench/types';

import { InfoIcon } from 'lucide-react';

import { VersionStatusWidgetView } from './VersionStatusWidgetView';

export const versionStatusWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['bottom'],
  bottomPanel: 'tooltip',
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: InfoIcon,
  id: 'version-status',
  label: 'Version',
  labelText: 'Version',
  version: 1,
  view: VersionStatusWidgetView,
};
