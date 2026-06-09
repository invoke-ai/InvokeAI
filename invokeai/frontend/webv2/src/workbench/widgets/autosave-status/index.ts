import type { WidgetManifest } from '../../types';
import { AutosaveStatusWidgetView } from './AutosaveStatusWidgetView';

export const autosaveStatusWidgetManifest: WidgetManifest = {
  bottomPanel: 'tooltip',
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: 'lucide-react:cloud-check',
  id: 'autosave-status',
  label: 'Autosave',
  labelText: 'Autosave',
  regions: ['bottom'],
  version: 1,
  view: AutosaveStatusWidgetView,
};
