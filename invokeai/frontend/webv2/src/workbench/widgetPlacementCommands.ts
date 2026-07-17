import type { Dispatch } from 'react';

import type {
  OpenWorkbenchWidgetOptions,
  RegisteredWidget,
  WidgetInstanceId,
  WidgetRegion,
  WidgetRegionState,
  WidgetTypeId,
  WidgetWorkbenchApiResult,
} from './types';
import type { WidgetDragEndResolution } from './widgetDnd';
import type { WidgetPlacementMeta } from './widgetRegionViewModel';
import type { WorkbenchAction } from './workbenchState';

import { flushWorkbenchDrafts } from './widgets/draftRegistry';

export interface WidgetPlacementProject {
  projectId?: string;
  widgetInstances: WidgetPlacementMeta;
  widgetRegions: Record<WidgetRegion, Pick<WidgetRegionState, 'activeInstanceId' | 'instanceIds'>>;
}

const DEFAULT_OPEN_REGIONS: ReadonlyArray<WidgetRegion> = ['center', 'right', 'left', 'bottom'];

export type WidgetPlacementCommandResult = WidgetWorkbenchApiResult;

export const canRenderWidgetInRegion = (widget: RegisteredWidget, region: WidgetRegion): boolean => {
  if (widget.status !== 'enabled' || !widget.manifest.view || !widget.manifest.allowedRegions.includes(region)) {
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
      return { region, widget };
    }
  }

  return null;
};

export const openWidgetPlacement = ({
  dispatch,
  getWidgetsForRegion,
  options,
  projectId,
  typeId,
}: {
  dispatch: Dispatch<WorkbenchAction>;
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[];
  projectId?: string;
  typeId: WidgetTypeId;
  options?: OpenWorkbenchWidgetOptions;
}): WidgetPlacementCommandResult => {
  const target = getOpenableWidgetPlacement({ getWidgetsForRegion, options, typeId });

  if (!target) {
    return { ok: false, reason: 'unavailable' };
  }

  dispatch({
    createNew: target.widget.manifest.allowMultiple ? options?.createNew : undefined,
    initialValues: target.widget.manifest.state?.createInitial(),
    projectId,
    region: target.region,
    type: 'openRegionWidget',
    widgetId: typeId,
  });

  return { ok: true, region: target.region };
};

const getEnabledCenterViewCount = (
  project: WidgetPlacementProject,
  getWidgetById: (typeId: WidgetTypeId) => RegisteredWidget | undefined
): number =>
  project.widgetRegions.center.instanceIds.filter((instanceId) => {
    const instance = project.widgetInstances[instanceId];
    const widget = instance ? getWidgetById(instance.typeId) : undefined;

    return widget?.status === 'enabled' && widget.manifest.centerPlacement !== 'toolbar';
  }).length;

export const closeWidgetPlacement = ({
  dispatch,
  getWidgetById,
  instanceId,
  project,
  region,
}: {
  dispatch: Dispatch<WorkbenchAction>;
  getWidgetById: (typeId: WidgetTypeId) => RegisteredWidget | undefined;
  project: WidgetPlacementProject;
  region: WidgetRegion;
  instanceId: WidgetInstanceId;
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
  dispatch({ projectId: project.projectId, region, type: 'toggleRegionWidget', widgetId: instanceId });

  return { ok: true, region };
};

export const revealWidgetPlacement = ({
  dispatch,
  instanceId,
  project,
  region,
}: {
  dispatch: Dispatch<WorkbenchAction>;
  project: WidgetPlacementProject;
  region: WidgetRegion;
  instanceId: WidgetInstanceId;
}): WidgetPlacementCommandResult => {
  if (!project.widgetInstances[instanceId]) {
    return { ok: false, reason: 'not-found' };
  }

  if (!project.widgetRegions[region].instanceIds.includes(instanceId)) {
    return { ok: false, reason: 'unsupported' };
  }

  dispatch({ projectId: project.projectId, region, type: 'selectRegionWidget', widgetId: instanceId });

  return { ok: true, region };
};

export const dispatchWidgetDragEndPlacement = ({
  dispatch,
  resolution,
}: {
  dispatch: Dispatch<WorkbenchAction>;
  resolution: WidgetDragEndResolution;
}): WidgetPlacementCommandResult => {
  if (resolution.type === 'reorder') {
    dispatch({
      activeInstanceId: resolution.activeInstanceId,
      instanceIds: resolution.instanceIds,
      region: resolution.region,
      type: 'reorderWidgetInstances',
    });

    return { ok: true, region: resolution.region };
  }

  dispatch({
    fromRegion: resolution.fromRegion,
    instanceId: resolution.instanceId,
    toIndex: resolution.toIndex,
    toRegion: resolution.toRegion,
    type: 'moveWidgetInstance',
  });

  return { ok: true, region: resolution.toRegion };
};
