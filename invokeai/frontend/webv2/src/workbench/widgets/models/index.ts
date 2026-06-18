import type { WidgetManifest } from '@workbench/types';

import { BoxIcon } from 'lucide-react';

import { MaintenanceMenu } from './MaintenanceMenu';
import { ModelsWidgetView } from './ModelsWidgetView';

export const modelsWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['left', 'right', 'center'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  headerActions: MaintenanceMenu,
  icon: BoxIcon,
  id: 'models',
  label: 'Models',
  labelText: 'Models',
  version: 1,
  view: ModelsWidgetView,
};
