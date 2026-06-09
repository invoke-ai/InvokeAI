import type { WidgetManifest } from '../../types';
import { QueueWidgetView } from './QueueWidgetView';

export const queueWidgetManifest: WidgetManifest = {
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: 'lucide-react:list-ordered',
  id: 'queue',
  label: 'Queue',
  labelText: 'Queue',
  regions: ['right', 'bottom'],
  version: 1,
  view: QueueWidgetView,
};
