import type { WidgetManifest } from '@workbench/types';

import { LayoutActionsWidgetView } from './LayoutActionsWidgetView';
import { LayoutHeaderActions } from './LayoutHeaderActions';

export const layoutActionsWidgetManifest: WidgetManifest = {
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  headerActions: LayoutHeaderActions,
  icon: 'lucide-react:panel-bottom',
  id: 'layout-actions',
  label: 'Layout Actions',
  labelText: 'Layout Actions',
  regions: ['bottom'],
  version: 1,
  view: LayoutActionsWidgetView,
};
