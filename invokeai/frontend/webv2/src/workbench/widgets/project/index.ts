import type { WidgetManifest } from '@workbench/types';

import { FolderCogIcon } from 'lucide-react';

import { ProjectWidgetView } from './ProjectWidgetView';

export const projectWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['left', 'right'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: FolderCogIcon,
  id: 'project',
  label: 'Project',
  labelText: 'Project',
  version: 1,
  view: ProjectWidgetView,
};
