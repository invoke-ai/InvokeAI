import type { WidgetManifest } from '@workbench/types';
import { GenerateWidgetView } from './GenerateWidgetView';

export const generateWidgetManifest: WidgetManifest = {
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  graphBearing: { defaultGraphId: 'generate-graph', sourceId: 'generate', surfaces: ['left'] },
  icon: 'lucide-react:sliders-horizontal',
  id: 'generate',
  label: 'Generate',
  labelText: 'Generate',
  regions: ['left'],
  version: 1,
  view: GenerateWidgetView,
};
