import type { WidgetManifest } from '@workbench/widgetContracts';

import { autosaveStatusWidgetManifest } from './autosave-status/manifest';
import { canvasWidgetManifest } from './canvas/manifest';
import { diagnosticsWidgetManifest } from './diagnostics/manifest';
import { galleryWidgetManifest } from './gallery/manifest';
import { generateWidgetManifest } from './generate/manifest';
import { imageMapWidgetManifest } from './image-map/manifest';
import { layersWidgetManifest } from './layers/manifest';
import { notificationsWidgetManifest } from './notifications/manifest';
import { previewWidgetManifest } from './preview/manifest';
import { projectWidgetManifest } from './project/manifest';
import { queueStatusWidgetManifest } from './queue-status/manifest';
import { queueWidgetManifest } from './queue/manifest';
import { serverStatusWidgetManifest } from './server-status/manifest';
import { upscaleWidgetManifest } from './upscale/manifest';
import { videoWidgetManifest } from './video/manifest';
import { workflowWidgetManifest } from './workflow/manifest';

export const firstPartyWidgetManifests: WidgetManifest[] = [
  generateWidgetManifest,
  workflowWidgetManifest,
  upscaleWidgetManifest,
  videoWidgetManifest,
  canvasWidgetManifest,
  diagnosticsWidgetManifest,
  galleryWidgetManifest,
  imageMapWidgetManifest,
  previewWidgetManifest,
  projectWidgetManifest,
  layersWidgetManifest,
  queueWidgetManifest,
  notificationsWidgetManifest,
  serverStatusWidgetManifest,
  queueStatusWidgetManifest,
  autosaveStatusWidgetManifest,
];
