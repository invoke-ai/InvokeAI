import type { WidgetRegion, WidgetRegionState } from '@workbench/layoutContracts';
import type {
  RegisteredWidget,
  WidgetInstanceId,
  WidgetTypeId,
  WidgetWorkbenchApiResult,
} from '@workbench/widgetContracts';

import { flushWorkbenchDrafts } from '@platform/react/draftRegistry';

import type { FloatingWidgetPlacement } from './floatingWindows';
import type { WidgetDragEndResolution } from './widgetDnd';
import type { WidgetPlacementMeta } from './widgetRegionViewModel';
import type { WorkbenchWidgetCommands } from './workbenchStore';
import type { LayoutAvailability, ResolvedLayout } from './workspace/resolveLayout';

import { takesLastCenterView } from './workspace/structureReducer';

export interface WidgetPlacementProject {
  /** Where each placed window returns to; absent means none float. Guest windows return nowhere and are not here. */
  floatingPlacements?: Record<WidgetInstanceId, FloatingWidgetPlacement>;
  /** What the layout shows that the workspace does not place (`ResolvedLayout.guests`). */
  guests?: ResolvedLayout['guests'];
  projectId?: string;
  widgetInstances: WidgetPlacementMeta;
  widgetRegions: Record<WidgetRegion, Pick<WidgetRegionState, 'activeInstanceId' | 'instanceIds'>>;
}

/** Where a structural add puts a widget when the control names no region. */
export const DEFAULT_ADD_REGIONS: readonly WidgetRegion[] = ['center', 'right', 'left', 'bottom'];

/** A widget that lives only as a status-bar chip: it renders no panel, so a region add refuses it. */
export const isStatusChipWidget = (widget: RegisteredWidget): boolean =>
  widget.manifest.allowedRegions.includes('bottom') &&
  (widget.manifest.bottomPanel === 'tooltip' || widget.manifest.bottomPanel === 'popover');

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

/**
 * Structural: adds the widget to the first of `regions` that renders it, or fronts the instance that region already
 * holds. Only the shell's add controls call it; navigation opens through the workspace resolver.
 */
export const addWidgetPlacement = ({
  createNew,
  getWidgetsForRegion,
  projectId,
  regions = DEFAULT_ADD_REGIONS,
  typeId,
  widgets,
}: {
  createNew?: boolean;
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[];
  projectId?: string;
  regions?: readonly WidgetRegion[];
  typeId: WidgetTypeId;
  widgets: WorkbenchWidgetCommands;
}): WidgetPlacementCommandResult => {
  for (const region of regions) {
    const widget = getWidgetsForRegion(region).find((candidate) => candidate.manifest.id === typeId);

    if (widget && canRenderWidgetInRegion(widget, region)) {
      widget.implementation.preload();
      widgets.open({
        createNew: widget.manifest.allowMultiple ? createNew : undefined,
        initialValues: widget.manifest.state?.createInitial(),
        projectId,
        region,
        widgetId: typeId,
      });

      return { ok: true, region };
    }
  }

  return { ok: false, reason: 'unavailable' };
};

/**
 * Structural: puts a widget in this workspace where it opens by default; a status chip goes back to the status bar.
 * The Widgets list's Add uses it.
 */
export const addWidgetToWorkspace = ({
  getWidgetsForRegion,
  widget,
  widgets,
}: {
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[];
  widget: RegisteredWidget;
  widgets: WorkbenchWidgetCommands;
}): boolean => {
  const { allowMultiple, id: typeId, state } = widget.manifest;

  if (addWidgetPlacement({ createNew: allowMultiple, getWidgetsForRegion, typeId, widgets }).ok) {
    return true;
  }

  if (widget.status !== 'enabled' || !isStatusChipWidget(widget)) {
    return false;
  }

  widgets.open({
    createNew: allowMultiple ? true : undefined,
    initialValues: state.createInitial(),
    region: 'bottom',
    widgetId: typeId,
  });

  return true;
};

export const closeWidgetPlacement = ({
  availability,
  instanceId,
  project,
  region,
  widgets,
}: {
  /** What the center's last view is read by (`takesLastCenterView`). */
  availability: Pick<LayoutAvailability, 'getPresentation'>;
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

  if (region === 'center' && takesLastCenterView(project, instanceId, availability)) {
    return { ok: false, reason: 'unsupported' };
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
 * Activate a rail slot. A docked tab is revealed in its region. A floating window's marker brings the window forward
 * and expands it instead: docking is the window's own control and the marker's menu.
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
 * Close a floating window outright — never dock-then-toggle, which would pop its panel open and retarget the route on
 * the way. Reports where the window would have returned, so focus can go somewhere instead of dying with the window.
 */
export const removeFloatingPlacement = ({
  instanceId,
  project,
  widgets,
}: {
  project: WidgetPlacementProject;
  instanceId: WidgetInstanceId;
  widgets: WorkbenchWidgetCommands;
}): { returnRegion: WidgetRegion } | null => {
  const placement = project.floatingPlacements?.[instanceId];

  if (!placement) {
    return null;
  }

  flushWorkbenchDrafts();
  widgets.closeFloating(instanceId);

  return { returnRegion: placement.returnRegion };
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
