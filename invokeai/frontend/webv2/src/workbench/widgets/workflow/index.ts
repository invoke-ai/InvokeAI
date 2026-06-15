import type { WidgetManifest } from '@workbench/types';
import { WorkflowHeaderActions, WorkflowMenuItems, WorkflowWidgetLabel } from './WorkflowWidgetChrome';
import { WorkflowWidgetView } from './WorkflowWidgetView';

export const workflowWidgetManifest: WidgetManifest = {
  bottomPanel: 'expandable',
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  graphBearing: {
    defaultGraphId: 'workflow-graph',
    sourceId: 'project-graph',
    surfaces: ['center', 'left', 'bottom'],
  },
  headerActions: WorkflowHeaderActions,
  headerMenu: WorkflowMenuItems,
  icon: 'lucide-react:workflow',
  id: 'workflow',
  label: WorkflowWidgetLabel,
  labelText: 'Workflow',
  regions: ['center', 'left', 'bottom'],
  settingsSection: 'workflow',
  version: 1,
  view: WorkflowWidgetView,
};
