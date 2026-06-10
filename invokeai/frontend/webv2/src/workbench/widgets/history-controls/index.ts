import type { WidgetManifest } from '../../types';
import { HistoryControlsWidgetView } from './HistoryControlsWidgetView';

export const historyControlsWidgetManifest: WidgetManifest = {
  centerPlacement: 'toolbar',
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: 'lucide-react:undo-2',
  id: 'history-controls',
  label: 'History Controls',
  labelText: 'History Controls',
  regions: ['bottom', 'center'],
  version: 1,
  view: HistoryControlsWidgetView,
};
