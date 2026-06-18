import type { WidgetManifest } from '@workbench/types';

import { Undo2Icon } from 'lucide-react';

import { HistoryControlsWidgetView } from './HistoryControlsWidgetView';

export const historyControlsWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['bottom'],
  centerPlacement: 'toolbar',
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: Undo2Icon,
  id: 'history-controls',
  label: 'History Controls',
  labelText: 'History Controls',
  version: 1,
  view: HistoryControlsWidgetView,
};
