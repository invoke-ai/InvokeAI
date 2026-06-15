import type { WidgetManifest } from '@workbench/types';
import { NotificationsHeaderActions } from './NotificationsHeaderActions';
import { NotificationsWidgetView } from './NotificationsWidgetView';

export const notificationsWidgetManifest: WidgetManifest = {
  bottomPanel: 'expandable',
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  headerActions: NotificationsHeaderActions,
  icon: 'lucide-react:bell',
  id: 'notifications',
  label: 'Notifications',
  labelText: 'Notifications',
  regions: ['bottom'],
  version: 1,
  view: NotificationsWidgetView,
};
