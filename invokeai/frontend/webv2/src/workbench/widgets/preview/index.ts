import type { WidgetManifest } from '@workbench/types';

import { EyeIcon } from 'lucide-react';

import { PreviewHeaderActions } from './PreviewHeaderActions';
import { PreviewWidgetLabel } from './PreviewWidgetChrome';
import { PreviewWidgetView } from './PreviewWidgetView';

export const previewWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['center', 'right'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  headerActions: PreviewHeaderActions,
  headerLabel: PreviewWidgetLabel,
  icon: EyeIcon,
  id: 'preview',
  label: (t) => t('widgets.labels.preview'),
  version: 1,
  view: PreviewWidgetView,
};
