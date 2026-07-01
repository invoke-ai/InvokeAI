import type { WidgetManifest } from '@workbench/types';

import { BellIcon } from 'lucide-react';

import { NotificationsHeaderActions } from './NotificationsHeaderActions';
import { NotificationsWidgetView } from './NotificationsWidgetView';

export const notificationsWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['bottom'],
  bottomPanel: 'expandable',
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  headerActions: NotificationsHeaderActions,
  icon: BellIcon,
  id: 'notifications',
  label: (t) => t('widgets.labels.notifications'),
  version: 1,
  view: NotificationsWidgetView,
};
