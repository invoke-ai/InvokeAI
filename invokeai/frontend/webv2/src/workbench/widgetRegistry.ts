import type { WidgetRegion } from '@workbench/layoutContracts';
import type {
  NormalizedWidgetManifest,
  RegisteredWidget,
  WidgetFailure,
  WidgetManifest,
  WidgetTypeId,
} from '@workbench/widgetContracts';

import { getAuthSession } from '@features/identity';

import { createDeferredResource } from './deferredResource';
import { createWidgetImplementationResource } from './widgetImplementationResource';
import { autosaveStatusWidgetManifest } from './widgets/autosave-status/manifest';
import { canvasWidgetManifest } from './widgets/canvas/manifest';
import { diagnosticsWidgetManifest } from './widgets/diagnostics/manifest';
import { galleryWidgetManifest } from './widgets/gallery/manifest';
import { generateWidgetManifest } from './widgets/generate/manifest';
import { imageMapWidgetManifest } from './widgets/image-map/manifest';
import { layersWidgetManifest } from './widgets/layers/manifest';
import { notificationsWidgetManifest } from './widgets/notifications/manifest';
import { previewWidgetManifest } from './widgets/preview/manifest';
import { projectWidgetManifest } from './widgets/project/manifest';
import { queueStatusWidgetManifest } from './widgets/queue-status/manifest';
import { queueWidgetManifest } from './widgets/queue/manifest';
import { serverStatusWidgetManifest } from './widgets/server-status/manifest';
import { upscaleWidgetManifest } from './widgets/upscale/manifest';
import { versionStatusWidgetManifest } from './widgets/version-status/manifest';
import { videoWidgetManifest } from './widgets/video/manifest';
import { workflowWidgetManifest } from './widgets/workflow/manifest';

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

    const host = manifest.loadHost ? createDeferredResource(manifest.loadHost) : undefined;

    try {
      validateManifest(manifest);

      return {
        host,
        implementation: createWidgetImplementationResource(manifest.id, manifest.load),
        manifest,
        status: 'enabled' as const,
      };
    } catch (error) {
      const failure = createFailure(manifest.id, error);
      const status = manifest.failurePolicy.onRegistrationFailure === 'hide' ? 'hidden' : 'disabled';

      return {
        failure,
        host,
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
    (widget) => widget.status === 'enabled' && widget.host !== undefined && isWidgetAvailable(widget)
  );

export const getWidgetById = (widgetId: WidgetTypeId): RegisteredWidget | undefined =>
  registeredWidgets.find((widget) => widget.manifest.id === widgetId);

/** Starts cached implementation loads. Unknown or disabled ids are skipped. */
export const warmWidgets = (typeIds: readonly WidgetTypeId[]): void => {
  for (const typeId of typeIds) {
    const widget = getWidgetById(typeId);

    if (widget?.status === 'enabled') {
      widget.implementation.preload();
    }
  }
};

/**
 * Like {@link warmWidgets}, but resolves when the implementations are in hand.
 *
 * Callers that are about to make widgets appear use this to get the modules
 * BEFORE committing the state change, so the render never suspends. Suspending
 * is not free even when the chunk is already downloaded: showing a fallback
 * makes React withhold the resolved content for `FALLBACK_THROTTLE_MS` (300ms)
 * to avoid a flash, which measured as the entire cost of a layout switch.
 */
export const loadWidgets = (typeIds: readonly WidgetTypeId[]): Promise<unknown> =>
  Promise.allSettled(
    typeIds.flatMap((typeId) => {
      const widget = getWidgetById(typeId);

      return widget?.status === 'enabled' ? [widget.implementation.load()] : [];
    })
  );

/** Whether every renderable widget in the set is already in memory. */
export const areWidgetsLoaded = (typeIds: readonly WidgetTypeId[]): boolean =>
  typeIds.every((typeId) => {
    const widget = getWidgetById(typeId);

    return widget?.status !== 'enabled' || widget.implementation.getStatus() === 'loaded';
  });

export const widgetRegistrationFailures = registeredWidgets.flatMap((widget) =>
  widget.failure ? [widget.failure] : []
);
