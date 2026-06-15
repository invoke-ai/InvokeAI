import type { RegisteredWidget, WidgetFailure, WidgetId, WidgetManifest, WorkbenchRegion } from './types';

import { getAuthSession } from './auth/session';
import { isSupportedIconId } from './iconResolver';
import { autosaveStatusWidgetManifest } from './widgets/autosave-status';
import { canvasWidgetManifest } from './widgets/canvas';
import { diagnosticsWidgetManifest } from './widgets/diagnostics';
import { galleryWidgetManifest } from './widgets/gallery';
import { generateWidgetManifest } from './widgets/generate';
import { historyControlsWidgetManifest } from './widgets/history-controls';
import { layersWidgetManifest } from './widgets/layers';
import { layoutActionsWidgetManifest } from './widgets/layout-actions';
import { modelsWidgetManifest } from './widgets/models';
import { notificationsWidgetManifest } from './widgets/notifications';
import { previewWidgetManifest } from './widgets/preview';
import { projectWidgetManifest } from './widgets/project';
import { queueWidgetManifest } from './widgets/queue';
import { serverStatusWidgetManifest } from './widgets/server-status';
import { versionStatusWidgetManifest } from './widgets/version-status';
import { workflowWidgetManifest } from './widgets/workflow';

const firstPartyWidgetManifests: WidgetManifest[] = [
  generateWidgetManifest,
  workflowWidgetManifest,
  canvasWidgetManifest,
  diagnosticsWidgetManifest,
  galleryWidgetManifest,
  previewWidgetManifest,
  projectWidgetManifest,
  layersWidgetManifest,
  modelsWidgetManifest,
  queueWidgetManifest,
  notificationsWidgetManifest,
  serverStatusWidgetManifest,
  autosaveStatusWidgetManifest,
  historyControlsWidgetManifest,
  layoutActionsWidgetManifest,
  versionStatusWidgetManifest,
];

const createFailure = (widgetId: WidgetId, error: unknown): WidgetFailure => ({
  details: error instanceof Error ? (error.stack ?? error.message) : String(error),
  message: error instanceof Error ? error.message : `Failed to register ${widgetId}.`,
  occurredAt: new Date().toISOString(),
  widgetId,
});

const renderableRegions = new Set<WorkbenchRegion>(['bottom', 'center', 'dialog', 'left', 'popover', 'right']);

const validateManifest = (manifest: WidgetManifest): void => {
  if (manifest.regions.length === 0) {
    throw new Error(`Widget ${manifest.id} must declare at least one allowed region.`);
  }

  if (!isSupportedIconId(manifest.icon)) {
    throw new Error(`Widget ${manifest.id} references unsupported icon id ${manifest.icon}.`);
  }

  if (manifest.regions.some((region) => renderableRegions.has(region)) && !manifest.view) {
    throw new Error(`Widget ${manifest.id} declares a renderable region but does not include manifest.view.`);
  }
};

export const registerFirstPartyWidgets = (): RegisteredWidget[] =>
  firstPartyWidgetManifests.map((manifest) => {
    try {
      validateManifest(manifest);

      return { manifest, status: 'enabled' as const };
    } catch (error) {
      const failure = createFailure(manifest.id, error);
      const status = manifest.failurePolicy.onRegistrationFailure === 'hide' ? 'hidden' : 'disabled';

      return { failure, manifest, status };
    }
  });

export const registeredWidgets = registerFirstPartyWidgets();

/**
 * Admin-only widgets are offered only while an admin is signed in to a
 * multi-user backend. The imperative session read is safe here: the route
 * guard resolves the session before the workbench mounts, and a user change
 * remounts the workbench route.
 */
const isWidgetAvailable = (widget: RegisteredWidget): boolean => {
  if (!widget.manifest.requiresAdmin) {
    return true;
  }

  const session = getAuthSession();

  return session.multiuserEnabled && session.user?.is_admin === true;
};

export const getWidgetsForRegion = (region: WorkbenchRegion): RegisteredWidget[] =>
  registeredWidgets.filter(
    (widget) => widget.status !== 'hidden' && widget.manifest.regions.includes(region) && isWidgetAvailable(widget)
  );

export const getWidgetById = (widgetId: WidgetId): RegisteredWidget | undefined =>
  registeredWidgets.find((widget) => widget.manifest.id === widgetId);

export const widgetRegistrationFailures = registeredWidgets.flatMap((widget) =>
  widget.failure ? [widget.failure] : []
);
