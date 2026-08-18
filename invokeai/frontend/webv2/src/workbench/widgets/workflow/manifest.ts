import type { WidgetManifest } from '@workbench/widgetContracts';

import { loadWorkflowWidgetHost, loadWorkflowWidgetImplementation } from '@features/workflow/widget';
import { WorkflowIcon } from 'lucide-react';

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
  icon: WorkflowIcon,
  id: 'workflow',
  label: (t) => t('widgets.labels.workflow'),
  load: loadWorkflowWidgetImplementation,
  loadHost: loadWorkflowWidgetHost,
  settingsSection: 'workflow',
  version: 1,
};
