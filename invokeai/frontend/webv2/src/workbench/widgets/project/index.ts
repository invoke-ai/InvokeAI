import type { WidgetManifest } from '@workbench/types';

import { FolderCogIcon } from 'lucide-react';

import { ProjectWidgetView } from './ProjectWidgetView';

export const projectWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['left', 'right'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: FolderCogIcon,
  id: 'project',
  label: (t) => t('widgets.labels.project'),
  version: 1,
  view: ProjectWidgetView,
};
