import type { WidgetManifest } from '@workbench/types';

import { WorkflowIcon } from 'lucide-react';

import {
  WorkflowDialogHost,
  WorkflowHeaderActions,
  WorkflowMenuItems,
  WorkflowWidgetLabel,
} from './WorkflowWidgetChrome';
import { WorkflowWidgetView } from './WorkflowWidgetView';

export const workflowWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['center', 'left', 'bottom'],
  bottomPanel: 'expandable',
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  graphBearing: {
    defaultGraphId: 'workflow-graph',
    sourceId: 'workflow',
    surfaces: ['center', 'left', 'bottom'],
  },
  host: WorkflowDialogHost,
  headerActions: WorkflowHeaderActions,
  headerMenu: WorkflowMenuItems,
  icon: WorkflowIcon,
  id: 'workflow',
  label: WorkflowWidgetLabel,
  labelText: 'Workflow',
  settingsSection: 'workflow',
  version: 1,
  view: WorkflowWidgetView,
};
