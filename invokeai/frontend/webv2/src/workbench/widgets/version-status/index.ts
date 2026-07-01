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
  label: (t) => t('widgets.labels.versionStatus'),
  version: 1,
  view: VersionStatusWidgetView,
};
