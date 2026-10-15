import type { WidgetManifest } from '@workbench/widgetContracts';

import { LayersIcon } from 'lucide-react';

import { COLOR_PANE_DEFAULTS, LAYER_EDITOR_PANE_DEFAULTS } from './panes/editorPaneLayout';

export const layersWidgetManifest: WidgetManifest = {
  allowMultiple: false,
  allowedRegions: ['right'],
  // The rail context menu still offers close/move; the panel body is the chrome.
  chrome: { header: 'hidden' },
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: LayersIcon,
  id: 'layers',
  label: (t) => t('widgets.labels.layers'),
  load: () => import('./implementation').then((module) => module.widgetImplementation),
  state: {
    createInitial: () => ({ colorPane: { ...COLOR_PANE_DEFAULTS }, editorPanes: { ...LAYER_EDITOR_PANE_DEFAULTS } }),
    persistence: 'project',
    version: 1,
  },
  version: 1,
};
