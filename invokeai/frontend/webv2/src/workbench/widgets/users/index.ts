import type { WidgetManifest } from '../../types';
import { UsersWidgetView } from './UsersWidgetView';

export const usersWidgetManifest: WidgetManifest = {
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: 'lucide-react:users',
  id: 'users',
  label: 'Users',
  labelText: 'Users',
  regions: ['center'],
  requiresAdmin: true,
  version: 1,
  view: UsersWidgetView,
};
