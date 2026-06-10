import type { WidgetManifest } from '../../types';
import { DiagnosticsHeaderActions } from './DiagnosticsHeaderActions';
import { DiagnosticsWidgetView } from './DiagnosticsWidgetView';

export const diagnosticsWidgetManifest: WidgetManifest = {
  bottomPanel: 'expandable',
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  headerActions: DiagnosticsHeaderActions,
  icon: 'lucide-react:bug',
  id: 'diagnostics',
  label: 'Diagnostics',
  labelText: 'Diagnostics',
  regions: ['bottom', 'right'],
  version: 1,
  view: DiagnosticsWidgetView,
};
