import type { WidgetManifest } from '@workbench/types';

import { PanelBottomIcon } from 'lucide-react';

import { LayoutActionsWidgetView } from './LayoutActionsWidgetView';
import { LayoutHeaderActions } from './LayoutHeaderActions';

export const layoutActionsWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['bottom'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  headerActions: LayoutHeaderActions,
  icon: PanelBottomIcon,
  id: 'layout-actions',
  label: 'Layout Actions',
  labelText: 'Layout Actions',
  version: 1,
  view: LayoutActionsWidgetView,
};
