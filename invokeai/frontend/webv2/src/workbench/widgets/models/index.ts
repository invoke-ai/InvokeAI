import type { WidgetManifest } from '@workbench/types';

import { MaintenanceMenu } from './MaintenanceMenu';
import { ModelsWidgetView } from './ModelsWidgetView';

export const modelsWidgetManifest: WidgetManifest = {
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  headerActions: MaintenanceMenu,
  icon: 'lucide-react:box',
  id: 'models',
  label: 'Models',
  labelText: 'Models',
  regions: ['left', 'right', 'center'],
  version: 1,
  view: ModelsWidgetView,
};
