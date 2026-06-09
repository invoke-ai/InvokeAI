import type { WidgetManifest } from '../../types';
import { VersionStatusWidgetView } from './VersionStatusWidgetView';

export const versionStatusWidgetManifest: WidgetManifest = {
  bottomPanel: 'tooltip',
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: 'lucide-react:info',
  id: 'version-status',
  label: 'Version',
  labelText: 'Version',
  regions: ['bottom'],
  version: 1,
  view: VersionStatusWidgetView,
};
