import type { WidgetManifest } from '@workbench/types';

import { LayersIcon } from 'lucide-react';

import { LayersHeaderActions } from './LayersHeaderActions';
import { LayersWidgetView } from './LayersWidgetView';

export const layersWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['right'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  headerActions: LayersHeaderActions,
  icon: LayersIcon,
  id: 'layers',
  label: 'Layers',
  labelText: 'Layers',
  version: 1,
  view: LayersWidgetView,
};
