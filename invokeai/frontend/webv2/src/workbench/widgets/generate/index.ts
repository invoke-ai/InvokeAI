import type { WidgetManifest } from '@workbench/types';

import { SlidersHorizontalIcon } from 'lucide-react';

import { GenerateWidgetView } from './GenerateWidgetView';

export const generateWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['left'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  graphBearing: { defaultGraphId: 'generate-graph', sourceId: 'generate', surfaces: ['left'] },
  icon: SlidersHorizontalIcon,
  id: 'generate',
  label: 'Generate',
  labelText: 'Generate',
  version: 1,
  view: GenerateWidgetView,
};
