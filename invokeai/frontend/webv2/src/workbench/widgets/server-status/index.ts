import type { WidgetManifest } from '../../types';
import { ServerStatusWidgetView } from './ServerStatusWidgetView';

export const serverStatusWidgetManifest: WidgetManifest = {
  bottomPanel: 'tooltip',
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: 'lucide-react:plug-zap',
  id: 'server-status',
  label: 'Server Status',
  labelText: 'Server Status',
  regions: ['bottom'],
  version: 1,
  view: ServerStatusWidgetView,
};
