import type {
  FloatingWidgetState,
  LayoutPresetId,
  LayoutPresetSnapshot,
  LayoutPresetSnapshotLayout,
  LayoutPresetWidgetInstanceSnapshot,
  WidgetRegion,
  WidgetRegionState,
} from '@workbench/layoutContracts';
import type { WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';

import { WIDGET_REGIONS } from '@workbench/layoutContracts';
import { resolveLayoutPresetId } from '@workbench/layoutPresets';

import type { WorkspaceStructure } from './structure';
import type { SurfaceAvailability, WorkspaceView, WorkspaceViews } from './view';

import { getRegionDefaults, isSamePlacement, normalizeStructure, SIDE_REGIONS } from './structure';
import { setFrontOverride, setRegionCollapsed, setRegionSize, setWindowView } from './view';
import { getRegionFront } from './workspaceDerivations';

/** A project's unsaved arrangement of a workspace it was not on, as stored before structure and view split. */
export interface StoredWorkingLayout {
  presetId: LayoutPresetId;
  snapshot: LayoutPresetSnapshot;
}

/** The layout fields a project stored before structure and view split, as the project loader validated them. */
export interface StoredProjectLayout {
  floatingWidgets?: Record<WidgetInstanceId, FloatingWidgetState>;
  layout: LayoutPresetSnapshotLayout;
  presetWorkingLayouts?: readonly StoredWorkingLayout[];
  widgetInstances: Readonly<Record<WidgetInstanceId, { title?: string; typeId: WidgetTypeId }>>;
  widgetRegions: Record<WidgetRegion, WidgetRegionState>;
  /** When the document was last saved, where known: an account adopts its most recently used project's arrangements. */
  storedAt?: string;
  /** Saved beside the layout while the project was held (`Project.heldLegacyLayout`); migration does not read it. */
  heldView?: HeldView;
}

/** What a held project showed: the workspace it was on, its views, and what it had still to tell. */
export interface HeldView {
  droppedArrangements?: readonly LayoutPresetId[];
  views: WorkspaceViews;
  workspaceId: LayoutPresetId;
}

export interface ProjectLayoutMigration {
  workspaceId: LayoutPresetId;
  /** Per workspace, what the project's arrangement overrides of the saved structure; the empty views are left out. */
  views: WorkspaceViews;
  /**
   * Arrangements that place widgets differently from their saved workspace. Adopted as the workspace, the project's view
   * of it is empty; dropped, the project keeps `views`, which hold only what still applies to the saved structure.
   */
  structures: Partial<Record<LayoutPresetId, WorkspaceStructure>>;
}

const toArrangement = (stored: StoredProjectLayout, presetId: LayoutPresetId): WorkspaceStructure => {
  const hasInstance = (instanceId: WidgetInstanceId) => Object.hasOwn(stored.widgetInstances, instanceId);
  const referenced = new Set([
    ...WIDGET_REGIONS.flatMap((region) => [
      stored.widgetRegions[region].activeInstanceId,
      ...stored.widgetRegions[region].instanceIds,
    ]),
    ...Object.keys(stored.floatingWidgets ?? {}),
  ]);
  const widgetInstances: Record<WidgetInstanceId, LayoutPresetWidgetInstanceSnapshot> = {};

  for (const instanceId of [...referenced].filter(hasInstance)) {
    const { title, typeId } = stored.widgetInstances[instanceId]!;

    widgetInstances[instanceId] = { id: instanceId, typeId, ...(title === undefined ? {} : { title }) };
  }

  return normalizeStructure({
    ...(stored.floatingWidgets ? { floatingWidgets: stored.floatingWidgets } : {}),
    layout: { ...stored.layout, presetId },
    widgetInstances,
    widgetRegions: stored.widgetRegions,
  });
};

/** What `arrangement` changes about `saved` that a view can hold. */
const extractView = (
  arrangement: WorkspaceStructure,
  saved: WorkspaceStructure,
  surfaces: SurfaceAvailability
): WorkspaceView | undefined => {
  let view: WorkspaceView | undefined;

  for (const region of WIDGET_REGIONS) {
    const front = getRegionFront(arrangement.widgetRegions[region]);

    if (front !== null) {
      view = setFrontOverride(view, saved, region, front, surfaces);
    }
  }
  for (const region of SIDE_REGIONS) {
    const defaults = getRegionDefaults(arrangement, region);

    view = setRegionSize(view, saved, region, defaults.sizePx);
    // A rail emptied by the arrangement is collapsed for being empty, which says nothing about the saved one.
    if (arrangement.widgetRegions[region].instanceIds.length > 0) {
      view = setRegionCollapsed(view, saved, region, defaults.collapsed);
    }
  }
  for (const [instanceId, { heightPx, mode, stackOrder, widthPx, x, y }] of Object.entries(
    arrangement.floatingWidgets ?? {}
  )) {
    view = setWindowView(view, saved, instanceId, { heightPx, mode, stackOrder, widthPx, x, y });
  }

  return view;
};

/**
 * Splits each arrangement a project stored against its workspace: the live one for the workspace it is on, and each
 * working copy for its own. Pure and total over validated shapes. A workspace the account no longer has gets no view;
 * a project on one moves to `defaultWorkspaceId`. Legacy preset ids are read as their current ones.
 */
export const migrateProjectLayout = (
  stored: StoredProjectLayout,
  {
    defaultWorkspaceId,
    surfaces,
    workspaces,
  }: {
    defaultWorkspaceId: LayoutPresetId;
    surfaces: SurfaceAvailability;
    /** The account's saved structures by workspace id: built-ins with their overrides, and custom ones. */
    workspaces: Readonly<Partial<Record<LayoutPresetId, WorkspaceStructure>>>;
  }
): ProjectLayoutMigration => {
  const hasWorkspace = (workspaceId: LayoutPresetId) => Object.hasOwn(workspaces, workspaceId);
  const activeId = resolveLayoutPresetId(stored.layout.presetId);
  const workspaceId = hasWorkspace(activeId) ? activeId : defaultWorkspaceId;
  const arrangements = new Map<LayoutPresetId, WorkspaceStructure>();

  // A later copy of one preset is the newer; the live layout outranks a copy of the preset it is on.
  for (const { presetId, snapshot } of stored.presetWorkingLayouts ?? []) {
    const copyId = resolveLayoutPresetId(presetId);

    if (copyId !== activeId && hasWorkspace(copyId)) {
      arrangements.set(copyId, normalizeStructure({ ...snapshot, layout: { ...snapshot.layout, presetId: copyId } }));
    }
  }
  if (workspaceId === activeId) {
    arrangements.set(activeId, toArrangement(stored, activeId));
  }

  const views: WorkspaceViews = {};
  const structures: ProjectLayoutMigration['structures'] = {};

  for (const [id, arrangement] of arrangements) {
    const saved = normalizeStructure(workspaces[id]!);
    const view = extractView(arrangement, saved, surfaces);

    if (view) {
      views[id] = view;
    }
    if (!isSamePlacement(arrangement, saved)) {
      structures[id] = arrangement;
    }
  }

  return { structures, views, workspaceId };
};
