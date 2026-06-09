import type { WidgetManifest } from '../../types';
import { WorkflowWidgetView } from './WorkflowWidgetView';

export const workflowWidgetManifest: WidgetManifest = {
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  graphBearing: {
    defaultGraphId: 'workflow-graph',
    sourceId: 'project-graph',
    surfaces: ['center', 'left'],
  },
  icon: 'lucide-react:workflow',
  id: 'workflow',
  label: 'Workflow',
  labelText: 'Workflow',
  regions: ['center', 'left'],
  version: 1,
  view: WorkflowWidgetView,
};
