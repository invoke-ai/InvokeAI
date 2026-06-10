import type { WidgetManifest } from '../../types';
import { PreviewWidgetView } from './PreviewWidgetView';

export const previewWidgetManifest: WidgetManifest = {
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: 'lucide-react:eye',
  id: 'preview',
  label: 'Preview',
  labelText: 'Preview',
  regions: ['center', 'right'],
  version: 1,
  view: PreviewWidgetView,
};
