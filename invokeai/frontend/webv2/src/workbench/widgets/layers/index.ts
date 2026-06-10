import type { WidgetManifest } from '../../types';
import { LayersHeaderActions } from './LayersHeaderActions';
import { LayersWidgetView } from './LayersWidgetView';

export const layersWidgetManifest: WidgetManifest = {
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  headerActions: LayersHeaderActions,
  icon: 'lucide-react:layers',
  id: 'layers',
  label: 'Layers',
  labelText: 'Layers',
  regions: ['right'],
  version: 1,
  view: LayersWidgetView,
};
