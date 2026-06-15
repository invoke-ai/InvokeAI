import type { WidgetManifest } from '@workbench/types';
import { ProjectWidgetView } from './ProjectWidgetView';

export const projectWidgetManifest: WidgetManifest = {
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: 'lucide-react:folder-cog',
  id: 'project',
  label: 'Project',
  labelText: 'Project',
  regions: ['left', 'right'],
  version: 1,
  view: ProjectWidgetView,
};
