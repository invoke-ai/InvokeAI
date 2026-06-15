import type { WidgetManifest } from '@workbench/types';

import { PreviewHeaderActions } from './PreviewHeaderActions';
import { PreviewWidgetView } from './PreviewWidgetView';

export const previewWidgetManifest: WidgetManifest = {
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  headerActions: PreviewHeaderActions,
  icon: 'lucide-react:eye',
  id: 'preview',
  label: 'Preview',
  labelText: 'Preview',
  regions: ['center', 'right'],
  version: 1,
  view: PreviewWidgetView,
};
