import type { WidgetManifest } from '@workbench/types';

import { CloudCheckIcon } from 'lucide-react';

import { AutosaveStatusWidgetView } from './AutosaveStatusWidgetView';

export const autosaveStatusWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['bottom'],
  bottomPanel: 'tooltip',
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: CloudCheckIcon,
  id: 'autosave-status',
  label: 'Autosave',
  labelText: 'Autosave',
  version: 1,
  view: AutosaveStatusWidgetView,
};
