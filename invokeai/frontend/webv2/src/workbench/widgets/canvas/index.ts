import type { WidgetManifest } from '@workbench/types';

import { WandSparklesIcon } from 'lucide-react';

import { CanvasWidgetView } from './CanvasWidgetView';

export const canvasWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['center'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  graphBearing: { defaultGraphId: 'canvas-graph', sourceId: 'canvas-fill', surfaces: ['center'] },
  icon: WandSparklesIcon,
  id: 'canvas',
  label: 'Canvas',
  labelText: 'Canvas',
  version: 1,
  view: CanvasWidgetView,
};
