import type { WidgetManifest } from '@workbench/types';

import { BugIcon } from 'lucide-react';

import { DiagnosticsHeaderActions } from './DiagnosticsHeaderActions';
import { DiagnosticsWidgetView } from './DiagnosticsWidgetView';

export const diagnosticsWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['bottom', 'right'],
  bottomPanel: 'expandable',
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  headerActions: DiagnosticsHeaderActions,
  icon: BugIcon,
  id: 'diagnostics',
  label: 'Diagnostics',
  labelText: 'Diagnostics',
  version: 1,
  view: DiagnosticsWidgetView,
};
