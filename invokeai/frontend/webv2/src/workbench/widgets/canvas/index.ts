import type { WidgetManifest } from '@workbench/types';
import { CanvasWidgetView } from './CanvasWidgetView';

export const canvasWidgetManifest: WidgetManifest = {
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  graphBearing: { defaultGraphId: 'canvas-graph', sourceId: 'canvas-fill', surfaces: ['center'] },
  icon: 'lucide-react:wand-sparkles',
  id: 'canvas',
  label: 'Canvas',
  labelText: 'Canvas',
  regions: ['center'],
  version: 1,
  view: CanvasWidgetView,
};
