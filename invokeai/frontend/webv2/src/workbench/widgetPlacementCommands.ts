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

import type { FloatingWidgetPlacement } from './floatingWindows';
import type { WidgetDragEndResolution } from './widgetDnd';
import type { WidgetPlacementMeta } from './widgetRegionViewModel';
import type { WorkbenchWidgetCommands } from './workbenchStore';

import { isAwaitedCenterView } from './floatingWindows';

export interface WidgetPlacementProject {
  /** Where each floating window returns to; absent means none float, as with `Project.floatingWidgets`. */
  floatingPlacements?: Record<WidgetInstanceId, FloatingWidgetPlacement>;
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

/** Cycle docked panel views in their displayed order, leaving windows and toolbar/popover controls alone. */
export const cycleRegionWidget = ({
  direction,
  getWidgetsForRegion,
  project,
  region,
  widgets,
}: {
  direction: -1 | 1;
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[];
  project: WidgetPlacementProject;
  region: WidgetRegion;
  widgets: WorkbenchWidgetCommands;
}): WidgetInstanceId | null => {
  const available = new Set(
    getWidgetsForRegion(region)
      .filter(
        (widget) =>
          canRenderWidgetInRegion(widget, region) && !(region === 'bottom' && widget.manifest.bottomPanel === 'popover')
      )
      .map((widget) => widget.manifest.id)
  );
  const state = project.widgetRegions[region];
  const ids = state.instanceIds.filter((id) => available.has(project.widgetInstances[id]?.typeId));
  if (ids.length === 0) {
    return null;
  }
  const current = ids.indexOf(state.activeInstanceId);
  const next =
    ids[current < 0 ? (direction > 0 ? 0 : ids.length - 1) : (current + direction + ids.length) % ids.length];
  // Selecting the same side-panel tab toggles its collapse state.
  if (next === state.activeInstanceId) {
    return null;
  }
  revealWidgetPlacement({ instanceId: next, project, region, widgets });
  return next;
};

/**
 * Activate a rail slot. A docked tab is revealed in its region. A floating window's marker brings the window
 * forward and expands it instead: docking is the window's own control and the marker's menu.
 */
export const activateRailPlacement = ({
  instanceId,
  project,
  region,
  widgets,
}: {
  project: WidgetPlacementProject;
  region: WidgetRegion;
  instanceId: WidgetInstanceId;
  widgets: WorkbenchWidgetCommands;
}): 'tab' | 'window' | null => {
  if (project.floatingPlacements?.[instanceId]) {
    widgets.revealFloating(instanceId);

    return 'window';
  }

  return revealWidgetPlacement({ instanceId, project, region, widgets }).ok ? 'tab' : null;
};

/** Dock a floating window where it came from; returns that region, or null when the instance does not float. */
export const dockFloatingPlacement = ({
  instanceId,
  project,
  widgets,
}: {
  project: WidgetPlacementProject;
  instanceId: WidgetInstanceId;
  widgets: WorkbenchWidgetCommands;
}): WidgetRegion | null => {
  const placement = project.floatingPlacements?.[instanceId];

  if (!placement) {
    return null;
  }

  // Docking remounts the widget; registry cleanup only removes flushers.
  flushWorkbenchDrafts();
  widgets.dockFloating(instanceId);

  return placement.returnRegion;
};

/**
 * Close a floating window outright — never dock-then-toggle, which would pop its panel open and retarget the route
 * on the way. Reports where the window would have returned and whether the reducer hands an emptied center its
 * view back, so focus can go somewhere instead of dying with the window.
 */
export const removeFloatingPlacement = ({
  instanceId,
  project,
  widgets,
}: {
  project: WidgetPlacementProject;
  instanceId: WidgetInstanceId;
  widgets: WorkbenchWidgetCommands;
}): { restoresCenter: boolean; returnRegion: WidgetRegion } | null => {
  const placement = project.floatingPlacements?.[instanceId];

  if (!placement) {
    return null;
  }

  const restoresCenter = isAwaitedCenterView(project.widgetRegions.center, instanceId);

  flushWorkbenchDrafts();
  widgets.closeFloating(instanceId);

  return { restoresCenter, returnRegion: placement.returnRegion };
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

  return { isPreviewActive, returnInstanceId };
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
 * Swaps the preview into the center and back to the view it replaced. Opening it in the center hands any rail
 * fronting the same instance to its neighbour (see `openRegionWidget`).
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
  const { isPreviewActive, returnInstanceId } = getCenterPreviewToggleState(project);

  if (isPreviewActive) {
    return (
      returnInstanceId !== null &&
      revealWidgetPlacement({ instanceId: returnInstanceId, project, region: 'center', widgets }).ok
    );
  }

  rememberReturnView(project);

  return openWidgetPlacement({
    getWidgetsForRegion,
    options: { preferredRegions: ['center'], requireCenterView: true },
    projectId: project.projectId,
    typeId: PREVIEW_TYPE_ID,
    widgets,
  }).ok;
};
