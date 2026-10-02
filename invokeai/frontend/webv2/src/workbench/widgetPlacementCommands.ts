import type { WidgetRegion, WidgetRegionState } from '@workbench/layoutContracts';
import type {
  OpenWorkbenchWidgetOptions,
  RegisteredWidget,
  WidgetInstanceId,
  WidgetTypeId,
  WidgetWorkbenchApiResult,
} from '@workbench/widgetContracts';

import { flushWorkbenchDrafts } from '@platform/react/draftRegistry';
import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

import type { WidgetDragEndResolution } from './widgetDnd';
import type { WidgetPlacementMeta } from './widgetRegionViewModel';
import type { WorkbenchWidgetCommands } from './workbenchStore';

export interface WidgetPlacementProject {
  projectId?: string;
  widgetInstances: WidgetPlacementMeta;
  widgetRegions: Record<WidgetRegion, Pick<WidgetRegionState, 'activeInstanceId' | 'instanceIds'>>;
}

const DEFAULT_OPEN_REGIONS: ReadonlyArray<WidgetRegion> = ['center', 'right', 'left', 'bottom'];

export type WidgetPlacementCommandResult = WidgetWorkbenchApiResult;

export const canRenderWidgetInRegion = (widget: RegisteredWidget, region: WidgetRegion): boolean => {
  if (widget.status !== 'enabled' || !widget.manifest.allowedRegions.includes(region)) {
    return false;
  }

  if (region === 'center') {
    return widget.manifest.centerPlacement !== 'toolbar';
  }

  if (region === 'bottom') {
    return widget.manifest.bottomPanel !== 'tooltip';
  }

  return true;
};

export const getOpenableWidgetPlacement = ({
  getWidgetsForRegion,
  options = {},
  typeId,
}: {
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[];
  typeId: WidgetTypeId;
  options?: OpenWorkbenchWidgetOptions;
}): { region: WidgetRegion; widget: RegisteredWidget } | null => {
  const preferredRegions = options.preferredRegions ?? DEFAULT_OPEN_REGIONS;
  const seenRegions = new Set<WidgetRegion>();

  for (const region of preferredRegions) {
    if (seenRegions.has(region) || (options.requireCenterView && region !== 'center')) {
      continue;
    }

    seenRegions.add(region);

    const widget = getWidgetsForRegion(region).find((candidate) => candidate.manifest.id === typeId);

    if (widget && canRenderWidgetInRegion(widget, region)) {
      widget.implementation.preload();
      return { region, widget };
    }
  }

  return null;
};

export const openWidgetPlacement = ({
  getWidgetsForRegion,
  options,
  projectId,
  typeId,
  widgets,
}: {
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[];
  projectId?: string;
  typeId: WidgetTypeId;
  widgets: WorkbenchWidgetCommands;
  options?: OpenWorkbenchWidgetOptions;
}): WidgetPlacementCommandResult => {
  const target = getOpenableWidgetPlacement({ getWidgetsForRegion, options, typeId });

  if (!target) {
    return { ok: false, reason: 'unavailable' };
  }

  widgets.open({
    createNew: target.widget.manifest.allowMultiple ? options?.createNew : undefined,
    initialValues: target.widget.manifest.state?.createInitial(),
    projectId,
    region: target.region,
    widgetId: typeId,
  });

  return { ok: true, region: target.region };
};

export const getEnabledCenterViewCount = (
  project: WidgetPlacementProject,
  getWidgetById: (typeId: WidgetTypeId) => RegisteredWidget | undefined
): number =>
  project.widgetRegions.center.instanceIds.filter((instanceId) => {
    const instance = project.widgetInstances[instanceId];
    const widget = instance ? getWidgetById(instance.typeId) : undefined;

    return widget?.status === 'enabled' && widget.manifest.centerPlacement !== 'toolbar';
  }).length;

export const closeWidgetPlacement = ({
  getWidgetById,
  instanceId,
  project,
  region,
  widgets,
}: {
  getWidgetById: (typeId: WidgetTypeId) => RegisteredWidget | undefined;
  project: WidgetPlacementProject;
  region: WidgetRegion;
  instanceId: WidgetInstanceId;
  widgets: WorkbenchWidgetCommands;
}): WidgetPlacementCommandResult => {
  if (!project.widgetInstances[instanceId]) {
    return { ok: false, reason: 'not-found' };
  }

  if (!project.widgetRegions[region].instanceIds.includes(instanceId)) {
    return { ok: false, reason: 'unsupported' };
  }

  if (region === 'center') {
    const instance = project.widgetInstances[instanceId];
    const widget = getWidgetById(instance.typeId);
    const isCenterView = widget?.manifest.centerPlacement !== 'toolbar';

    if (isCenterView && getEnabledCenterViewCount(project, getWidgetById) === 1) {
      return { ok: false, reason: 'unsupported' };
    }
  }

  flushWorkbenchDrafts();
  widgets.toggle({ projectId: project.projectId, region, widgetId: instanceId });

  return { ok: true, region };
};

export const revealWidgetPlacement = ({
  instanceId,
  project,
  region,
  widgets,
}: {
  project: WidgetPlacementProject;
  region: WidgetRegion;
  instanceId: WidgetInstanceId;
  widgets: WorkbenchWidgetCommands;
}): WidgetPlacementCommandResult => {
  if (!project.widgetInstances[instanceId]) {
    return { ok: false, reason: 'not-found' };
  }

  if (!project.widgetRegions[region].instanceIds.includes(instanceId)) {
    return { ok: false, reason: 'unsupported' };
  }

  widgets.select({ projectId: project.projectId, region, widgetId: instanceId });

  return { ok: true, region };
};

export const dispatchWidgetDragEndPlacement = ({
  resolution,
  widgets,
}: {
  resolution: WidgetDragEndResolution;
  widgets: WorkbenchWidgetCommands;
}): WidgetPlacementCommandResult => {
  if (resolution.type === 'reorder') {
    widgets.reorder({
      activeInstanceId: resolution.activeInstanceId,
      instanceIds: resolution.instanceIds,
      region: resolution.region,
    });

    if (resolution.align) {
      widgets.setAlignment({
        align: resolution.align,
        instanceId: resolution.activeInstanceId,
        region: resolution.region,
      });
    }

    return { ok: true, region: resolution.region };
  }

  widgets.move({
    fromRegion: resolution.fromRegion,
    instanceId: resolution.instanceId,
    toIndex: resolution.toIndex,
    toRegion: resolution.toRegion,
  });

  if (resolution.align) {
    widgets.setAlignment({ align: resolution.align, instanceId: resolution.instanceId, region: resolution.toRegion });
  }

  return { ok: true, region: resolution.toRegion };
};

const PREVIEW_TYPE_ID = 'preview';

/** The center view each project showed before the preview was swapped in; session-lived. */
const returnStore = createExternalStore<{ byProject: Record<string, WidgetInstanceId> }>({ byProject: {} });

registerAccountOwnedResource({
  clear: () => returnStore.setSnapshot({ byProject: {} }),
  name: 'center-preview-toggle',
});

export interface CenterPreviewToggleState {
  isPreviewActive: boolean;
  previewInstanceId: WidgetInstanceId | null;
  returnInstanceId: WidgetInstanceId | null;
}

export const getCenterPreviewToggleState = (project: WidgetPlacementProject): CenterPreviewToggleState => {
  const center = project.widgetRegions.center;
  const previewInstanceId =
    center.instanceIds.find((id) => project.widgetInstances[id]?.typeId === PREVIEW_TYPE_ID) ?? null;
  const isPreviewActive = previewInstanceId !== null && center.activeInstanceId === previewInstanceId;
  const remembered = project.projectId ? returnStore.getSnapshot().byProject[project.projectId] : undefined;
  const returnInstanceId =
    remembered && remembered !== previewInstanceId && center.instanceIds.includes(remembered)
      ? remembered
      : (center.instanceIds.find((id) => id !== previewInstanceId) ?? null);

  return { isPreviewActive, previewInstanceId, returnInstanceId };
};

const rememberReturnView = (project: WidgetPlacementProject): void => {
  if (!project.projectId) {
    return;
  }

  const { byProject } = returnStore.getSnapshot();

  returnStore.setSnapshot({
    byProject: { ...byProject, [project.projectId]: project.widgetRegions.center.activeInstanceId },
  });
};

/**
 * Swaps the preview into the center and back to the view it replaced. One preview instance can
 * sit in the center and a rail at once, so a rail actively showing it moves to its neighbour first.
 */
export const toggleCenterPreview = ({
  getWidgetsForRegion,
  project,
  widgets,
}: {
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[];
  project: WidgetPlacementProject;
  widgets: WorkbenchWidgetCommands;
}): boolean => {
  const { isPreviewActive, previewInstanceId, returnInstanceId } = getCenterPreviewToggleState(project);

  if (isPreviewActive) {
    return (
      returnInstanceId !== null &&
      revealWidgetPlacement({ instanceId: returnInstanceId, project, region: 'center', widgets }).ok
    );
  }

  rememberReturnView(project);

  for (const [region, state] of Object.entries(project.widgetRegions) as [
    WidgetRegion,
    { activeInstanceId: string; instanceIds: string[] },
  ][]) {
    if (region === 'center' || state.activeInstanceId !== previewInstanceId) {
      continue;
    }

    const neighbour = state.instanceIds.find((id) => id !== previewInstanceId);

    if (neighbour) {
      widgets.select({ projectId: project.projectId, region, widgetId: neighbour });
    }
  }

  return openWidgetPlacement({
    getWidgetsForRegion,
    options: { preferredRegions: ['center'], requireCenterView: true },
    projectId: project.projectId,
    typeId: PREVIEW_TYPE_ID,
    widgets,
  }).ok;
};
