import type { WidgetRegion } from '@workbench/layoutContracts';
import type {
  NormalizedWidgetManifest,
  RegisteredWidget,
  WidgetFailure,
  WidgetManifest,
  WidgetTypeId,
} from '@workbench/widgetContracts';

import { getAuthSession } from '@features/identity';
import { queueWidgetManifest } from '@workbench/queueWidgetManifest';

import { galleryWidgetManifest } from './galleryWidgetManifest';
import { generateWidgetManifest } from './generationWidgetManifest';
import { upscaleWidgetManifest } from './upscaleWidgetManifest';
import { createWidgetImplementationResource } from './widgetImplementationResource';
import { autosaveStatusWidgetManifest } from './widgets/autosave-status';
import { canvasWidgetManifest } from './widgets/canvas';
import { diagnosticsWidgetManifest } from './widgets/diagnostics';
import { layersWidgetManifest } from './widgets/layers';
import { notificationsWidgetManifest } from './widgets/notifications';
import { previewWidgetManifest } from './widgets/preview';
import { projectWidgetManifest } from './widgets/project';
import { serverStatusWidgetManifest } from './widgets/server-status';
import { versionStatusWidgetManifest } from './widgets/version-status';
import { workflowWidgetManifest } from './workflowWidgetManifest';

export const firstPartyWidgetManifests: WidgetManifest[] = [
  generateWidgetManifest,
  workflowWidgetManifest,
  upscaleWidgetManifest,
  canvasWidgetManifest,
  diagnosticsWidgetManifest,
  galleryWidgetManifest,
  previewWidgetManifest,
  projectWidgetManifest,
  layersWidgetManifest,
  queueWidgetManifest,
  notificationsWidgetManifest,
  serverStatusWidgetManifest,
  autosaveStatusWidgetManifest,
  versionStatusWidgetManifest,
];

const createFailure = (widgetId: WidgetTypeId, error: unknown): WidgetFailure => ({
  details: error instanceof Error ? (error.stack ?? error.message) : String(error),
  message: error instanceof Error ? error.message : `Failed to register ${widgetId}.`,
  occurredAt: new Date().toISOString(),
  widgetId,
});

const renderableRegions = new Set<WidgetRegion>(['bottom', 'center', 'left', 'right']);

const isWidgetIconComponent = (value: WidgetManifest['icon']): boolean =>
  typeof value === 'function' || (typeof value === 'object' && value !== null && '$$typeof' in value);

const validateManifest = (manifest: NormalizedWidgetManifest): void => {
  if (typeof manifest.id !== 'string' || manifest.id.trim().length === 0 || /\s/.test(manifest.id)) {
    throw new Error('Widget manifest must provide a stable non-empty string id without whitespace.');
  }

  if (manifest.apiVersion !== 1) {
    throw new Error(`Widget ${manifest.id} declares unsupported apiVersion ${String(manifest.apiVersion)}.`);
  }

  if (manifest.allowedRegions.length === 0) {
    throw new Error(`Widget ${manifest.id} must declare at least one allowed region.`);
  }

  for (const region of manifest.allowedRegions) {
    if (!renderableRegions.has(region)) {
      throw new Error(`Widget ${manifest.id} declares unsupported region ${String(region)}.`);
    }
  }

  if (!isWidgetIconComponent(manifest.icon)) {
    throw new TypeError(`Widget ${manifest.id} must provide an icon component.`);
  }

  if (typeof manifest.load !== 'function') {
    throw new TypeError(`Widget ${manifest.id} must provide a deferred implementation loader.`);
  }
};

export const normalizeWidgetManifest = (manifest: WidgetManifest): NormalizedWidgetManifest => ({
  ...manifest,
  apiVersion: manifest.apiVersion ?? 1,
  state: manifest.state ?? { createInitial: () => ({}), persistence: 'project', version: 1 },
});

export const registerWidgets = (manifests: WidgetManifest[]): RegisteredWidget[] =>
  manifests.map((rawManifest) => {
    const manifest = normalizeWidgetManifest(rawManifest);

    try {
      validateManifest(manifest);

      return {
        implementation: createWidgetImplementationResource(manifest.id, manifest.load),
        manifest,
        status: 'enabled' as const,
      };
    } catch (error) {
      const failure = createFailure(manifest.id, error);
      const status = manifest.failurePolicy.onRegistrationFailure === 'hide' ? 'hidden' : 'disabled';

      return {
        failure,
        implementation: createWidgetImplementationResource(manifest.id, manifest.load),
        manifest,
        status,
      };
    }
  });

export const registerFirstPartyWidgets = (): RegisteredWidget[] => registerWidgets(firstPartyWidgetManifests);

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

export const getWidgetsForRegion = (region: WidgetRegion): RegisteredWidget[] =>
  registeredWidgets.filter(
    (widget) =>
      widget.status !== 'hidden' && widget.manifest.allowedRegions.includes(region) && isWidgetAvailable(widget)
  );

export const getWidgetHosts = (): RegisteredWidget[] =>
  registeredWidgets.filter(
    (widget) => widget.status === 'enabled' && widget.manifest.hasHost && isWidgetAvailable(widget)
  );

export const getWidgetById = (widgetId: WidgetTypeId): RegisteredWidget | undefined =>
  registeredWidgets.find((widget) => widget.manifest.id === widgetId);

/** Starts a cached implementation load for an explicit user intent. */
export const preloadWidget = (widgetId: WidgetTypeId): void => {
  const widget = getWidgetById(widgetId);

  if (widget?.status === 'enabled') {
    widget.implementation.preload();
  }
};

export const widgetRegistrationFailures = registeredWidgets.flatMap((widget) =>
  widget.failure ? [widget.failure] : []
);
