import type { WidgetManifest } from '@workbench/widgetContracts';

import { loadWorkflowWidgetImplementation } from '@features/workflow/widget';
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
  hasHost: true,
  icon: WorkflowIcon,
  id: 'workflow',
  label: (t) => t('widgets.labels.workflow'),
  load: loadWorkflowWidgetImplementation,
  settingsSection: 'workflow',
  version: 1,
};
