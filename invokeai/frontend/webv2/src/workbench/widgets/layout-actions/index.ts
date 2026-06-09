import type { WidgetManifest } from '../../types';
import { LayoutActionsWidgetView } from './LayoutActionsWidgetView';

export const layoutActionsWidgetManifest: WidgetManifest = {
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: 'lucide-react:panel-bottom',
  id: 'layout-actions',
  label: 'Layout Actions',
  labelText: 'Layout Actions',
  regions: ['bottom'],
  version: 1,
  view: LayoutActionsWidgetView,
};
