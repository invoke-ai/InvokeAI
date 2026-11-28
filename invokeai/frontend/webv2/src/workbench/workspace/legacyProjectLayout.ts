import type {
  CenterViewId,
  LayoutPresetId,
  LayoutPresetSnapshotLayout,
  WidgetRegion,
  WidgetRegionState,
} from '@workbench/layoutContracts';
import type { AccountState, Project } from '@workbench/projectContracts';
import type { WidgetId, WidgetInstanceContract, WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';

import { normalizeFloatingPlacement } from '@workbench/floatingWindows';
import { defaultLayoutPreset, getLayoutPreset, layoutPresets, resolveLayoutPresetId } from '@workbench/layoutPresets';
import { cloneLayoutPresetWidgetRegions, doesLayoutPresetExist } from '@workbench/layoutPresetSnapshots';

import type { HeldView, StoredProjectLayout, StoredWorkingLayout } from './migrateProjectLayout';
import type { WorkspaceStructure } from './structure';
import type { SurfaceAvailability } from './view';

import { getAccountStructures, writeWorkspaceStructure } from './accountWorkspaces';
import { migrateProjectLayout } from './migrateProjectLayout';
import { digestStoredLayout } from './storedLayoutDigest';
import { isStoredStructure, readStoredStructure, stripRetiredWidgets } from './storedStructure';
import { isSamePlacement } from './structure';
import { normalizeViews } from './view';

/** The layout fields a document stored before structure and view split; unknown until validated. */
export interface StoredLayoutFields {
  floatingWidgets?: unknown;
  /** Not from before the split: what a held project showed (`StoredProjectLayout.heldView`), saved beside the rest. */
  heldView?: unknown;
  /** Not from before the split: the document came in through an import (`StoredProjectLayout.imported`). */
  importedLayout?: unknown;
  layout?: unknown;
  presetWorkingLayouts?: unknown;
  widgetRegions?: unknown;
}

const createWidgetRegions = (): Record<WidgetRegion, WidgetRegionState> => ({
  ...cloneLayoutPresetWidgetRegions(defaultLayoutPreset.snapshot.widgetRegions),
});

const LEGACY_DEFAULT_LEFT_REGION_WIDGET_IDS: readonly WidgetInstanceId[][] = [
  ['generate', 'workflow'],
  ['workflow', 'generate'],
  ['generate', 'workflow', 'gallery'],
];

const ensureLeftRegion = (leftRegion: WidgetRegionState): WidgetRegionState => {
  let region = leftRegion;

  if (!region.instanceIds.includes('upscale')) {
    const legacyMatch = LEGACY_DEFAULT_LEFT_REGION_WIDGET_IDS.some(
      (ids) => ids.length === region.instanceIds.length && ids.every((id, index) => region.instanceIds[index] === id)
    );

    if (legacyMatch) {
      const galleryIndex = region.instanceIds.indexOf('gallery');
      const instanceIds = [...region.instanceIds];

      instanceIds.splice(galleryIndex === -1 ? instanceIds.length : galleryIndex, 0, 'upscale');
      region = { ...region, instanceIds };
    }
  }

  return region;
};

// Exact historical defaults adopt the current rail wholesale; compared after retired widgets are stripped.
const LEGACY_RIGHT_REGION_WIDGET_IDS: WidgetId[][] = [
  ['queue', 'gallery', 'layers'],
  ['gallery', 'preview', 'queue', 'layers', 'diagnostics'],
];

const isLegacyDefaultRightRegion = (region: WidgetRegionState): boolean =>
  LEGACY_RIGHT_REGION_WIDGET_IDS.some(
    (ids) =>
      ids.length === region.instanceIds.length && ids.every((widgetId, index) => region.instanceIds[index] === widgetId)
  );

const ensureRightRegion = (rightRegion: WidgetRegionState): WidgetRegionState =>
  isLegacyDefaultRightRegion(rightRegion)
    ? { ...rightRegion, instanceIds: createWidgetRegions().right.instanceIds }
    : rightRegion;

/** Historical Edit rails, compared after the retired canvas editors beside Layers are stripped. */
const LEGACY_EDIT_RIGHT_REGION_WIDGET_IDS: ReadonlyArray<readonly WidgetInstanceId[]> = [
  ['layers', 'preview', 'gallery', 'image-map', 'queue'],
  ['layers', 'preview', 'gallery', 'queue'],
  ['layers'],
];

const sameInstanceIds = (region: WidgetRegionState, ids: readonly WidgetInstanceId[]): boolean =>
  region.instanceIds.length === ids.length && region.instanceIds.every((id, index) => id === ids[index]);

const ensureEditRightRegion = (right: WidgetRegionState): WidgetRegionState => {
  if (!LEGACY_EDIT_RIGHT_REGION_WIDGET_IDS.some((ids) => sameInstanceIds(right, ids))) {
    return right;
  }
  const edit = getLayoutPreset('edit').snapshot.widgetRegions.right;
  return { ...right, activeInstanceId: edit.activeInstanceId, instanceIds: [...edit.instanceIds] };
};

/** Retired widget types; anything persisted about them drops on load. */

const LEGACY_DEFAULT_BOTTOM_REGION_WIDGET_IDS: readonly WidgetInstanceId[] = [
  'server-status',
  'gallery:bottom',
  'notifications',
  'autosave-status',
];

const isLegacyDefaultBottomRegion = (region: WidgetRegionState): boolean =>
  region.instanceIds.length === LEGACY_DEFAULT_BOTTOM_REGION_WIDGET_IDS.length &&
  region.instanceIds.every((widgetId, index) => widgetId === LEGACY_DEFAULT_BOTTOM_REGION_WIDGET_IDS[index]);

const ensureBottomRegion = (bottomRegion: WidgetRegionState): WidgetRegionState => {
  if (bottomRegion.instanceIds.includes('queue-status')) {
    return bottomRegion;
  }

  if (!isLegacyDefaultBottomRegion(bottomRegion)) {
    return bottomRegion;
  }

  const serverStatusIndex = bottomRegion.instanceIds.indexOf('server-status');
  const instanceIds = [...bottomRegion.instanceIds];

  instanceIds.splice(serverStatusIndex === -1 ? instanceIds.length : serverStatusIndex + 1, 0, 'queue-status');

  return { ...bottomRegion, instanceIds };
};

const getCenterWidgetIdFromViewId = (centerViewId: CenterViewId): WidgetInstanceId => {
  if (centerViewId === 'gallery') {
    return 'gallery:center';
  }

  if (centerViewId === 'workflow') {
    return 'workflow:center';
  }

  return centerViewId;
};

const ensureCenterRegion = (
  centerRegion: WidgetRegionState | undefined,
  fallbackCenterViewId: CenterViewId
): WidgetRegionState => {
  const defaultCenterRegion = createWidgetRegions().center;

  return {
    ...defaultCenterRegion,
    ...centerRegion,
    activeInstanceId: centerRegion?.activeInstanceId ?? getCenterWidgetIdFromViewId(fallbackCenterViewId),
    instanceIds: centerRegion ? centerRegion.instanceIds : defaultCenterRegion.instanceIds,
    isCollapsed: false,
  };
};

export const isRecordValue = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/**
 * A stored arrangement read the way this build read it before the split: retired widgets stripped, historical default
 * rails brought up to date, windows normalized; the instances it docks without values get them.
 */
export const loadStoredProjectLayout = (
  stored: Required<Omit<StoredLayoutFields, 'heldView' | 'importedLayout'>> &
    Pick<StoredLayoutFields, 'heldView' | 'importedLayout'> & {
      layout: Record<string, unknown>;
    },
  instances: Record<WidgetInstanceId, WidgetInstanceContract>,
  createInstance: (typeId: WidgetTypeId) => WidgetInstanceContract
): { legacyLayout: StoredProjectLayout; widgetInstances: Record<WidgetInstanceId, WidgetInstanceContract> } => {
  const layout = stored.layout as Partial<LayoutPresetSnapshotLayout>;
  const legacyWidgetRegions = stored.widgetRegions as
    | Partial<Record<WidgetRegion | 'left-panel' | 'right-panel' | 'status-bar', WidgetRegionState>>
    | undefined;
  const defaultRegions = createWidgetRegions();
  // Strip retired widgets first so normalization stays idempotent.
  const stripped = stripRetiredWidgets(
    {
      left: legacyWidgetRegions?.left ?? legacyWidgetRegions?.['left-panel'] ?? defaultRegions.left,
      right: legacyWidgetRegions?.right ?? legacyWidgetRegions?.['right-panel'] ?? defaultRegions.right,
      bottom: legacyWidgetRegions?.bottom ?? legacyWidgetRegions?.['status-bar'] ?? defaultRegions.bottom,
      center: ensureCenterRegion(legacyWidgetRegions?.center, layout.centerViewId ?? 'canvas'),
    },
    instances,
    stored.floatingWidgets
  );
  const leftRegion = ensureLeftRegion(stripped.widgetRegions.left);
  const bottomRegion = ensureBottomRegion(stripped.widgetRegions.bottom);
  const widgetInstances = stripped.widgetInstances;

  for (const [instanceId, region] of [
    ['upscale', leftRegion],
    ['video', leftRegion],
    ['queue-status', bottomRegion],
  ] as const) {
    if (region.instanceIds.includes(instanceId) && !widgetInstances[instanceId]) {
      widgetInstances[instanceId] = createInstance(instanceId);
    }
  }

  const placement = normalizeFloatingPlacement(
    {
      ...stripped.widgetRegions,
      bottom: bottomRegion,
      left: leftRegion,
      right: ensureEditRightRegion(ensureRightRegion(stripped.widgetRegions.right)),
    },
    stored.floatingWidgets,
    (instanceId) => Object.hasOwn(widgetInstances, instanceId)
  );
  const presetId = resolveLayoutPresetId(
    typeof layout.presetId === 'string' ? layout.presetId : defaultLayoutPreset.id
  );
  const presetWorkingLayouts = normalizePresetWorkingLayouts(stored.presetWorkingLayouts, presetId);
  const heldView = normalizeHeldView(stored.heldView, digestStoredLayout(stored));

  return {
    legacyLayout: {
      ...(placement.floatingWidgets ? { floatingWidgets: placement.floatingWidgets } : {}),
      ...(heldView ? { heldView } : {}),
      ...(stored.importedLayout === true ? { imported: true } : {}),
      layout: {
        centerViewId: layout.centerViewId ?? defaultLayoutPreset.snapshot.layout.centerViewId,
        panels: { ...defaultLayoutPreset.snapshot.layout.panels, ...layout.panels },
        presetId,
      },
      ...(presetWorkingLayouts ? { presetWorkingLayouts } : {}),
      widgetInstances: Object.fromEntries(
        Object.values(widgetInstances).map(({ id, title, typeId }) => [
          id,
          { typeId, ...(title === undefined ? {} : { title }) },
        ])
      ),
      widgetRegions: placement.widgetRegions,
    },
    widgetInstances,
  };
};

/** A held project's saved view, read as untrusted data, and only beside the layout it was saved with. */
const normalizeHeldView = (stored: unknown, layoutDigest: string): HeldView | undefined => {
  if (
    !isRecordValue(stored) ||
    stored.layoutDigest !== layoutDigest ||
    typeof stored.workspaceId !== 'string' ||
    stored.workspaceId.length === 0
  ) {
    return undefined;
  }

  const readIds = (value: unknown) =>
    Array.isArray(value)
      ? [...new Set(value.filter((id): id is string => typeof id === 'string' && id.length > 0))]
      : [];
  const dropped = readIds(stored.dropped);
  const untold = readIds(stored.untold);

  return {
    ...(stored.adopted === true ? { adopted: true } : {}),
    ...(dropped.length > 0 ? { dropped } : {}),
    ...(untold.length > 0 ? { untold } : {}),
    views: normalizeViews(stored.views, () => true),
    workspaceId: resolveLayoutPresetId(stored.workspaceId),
  };
};

/** A stored project's working copies, validated as stored workspaces are; the active workspace's is the live one. */
const normalizePresetWorkingLayouts = (
  stored: unknown,
  activePresetId: LayoutPresetId
): StoredWorkingLayout[] | undefined => {
  if (!Array.isArray(stored)) {
    return undefined;
  }

  const copies: StoredWorkingLayout[] = [];

  for (const entry of stored as unknown[]) {
    const { presetId: storedId, snapshot } = (entry ?? {}) as Partial<Record<keyof StoredWorkingLayout, unknown>>;
    const presetId = typeof storedId === 'string' ? resolveLayoutPresetId(storedId) : '';

    if (!presetId || presetId === activePresetId || !isStoredStructure(snapshot)) {
      continue;
    }

    const clone = readStoredStructure(snapshot);
    // A historical id and its current one name the same preset; the later entry is the newer copy.
    const earlier = copies.findIndex((copy) => copy.presetId === presetId);

    if (earlier !== -1) {
      copies.splice(earlier, 1);
    }
    copies.push({ presetId, snapshot: { ...clone, layout: { ...clone.layout, presetId } } });
  }

  return copies.length > 0 ? copies : undefined;
};

type StructuresById = Readonly<Partial<Record<LayoutPresetId, WorkspaceStructure>>>;

const SHIPPED_STRUCTURES: StructuresById = Object.fromEntries(
  layoutPresets.map((preset) => [preset.id, preset.snapshot])
);

/**
 * Whether adoption may make `arrangement` the workspace's structure: never over one the workspace model wrote, and not
 * for an arrangement that places a built-in as it ships, which is no change of the user's.
 */
const isAdoptable = (account: AccountState, workspaceId: LayoutPresetId, arrangement: WorkspaceStructure): boolean => {
  const shipped = SHIPPED_STRUCTURES[workspaceId];

  return (
    !account.writtenStructureIds?.includes(workspaceId) &&
    (shipped === undefined || !isSamePlacement(arrangement, shipped))
  );
};

/**
 * A loaded project settled against the account: its stored layout migrates, and each structurally different
 * arrangement is adopted as its workspace (`adopt`, where `isAdoptable`) or dropped. Dropped are only arrangements that
 * differ from every baseline as well as from the account (the workspace as the project last saw it, as it ships): one
 * that matches is no arrangement of the project's own. What the project is told follows from this load alone: a drop
 * its held view says the user was already told about is not told again. `announce` tells it which workspaces took its
 * arrangements. `hold` keeps the stored layout to be written back until the account's adoption is saved, with what
 * this load decided; the view it was held with, saved beside it, is the project's again. A workspace the account lacks
 * gives way to the default one, and views of workspaces it lacks go.
 */
const settleProjectLayout = (
  account: AccountState,
  project: Project,
  {
    adopt = false,
    announce = false,
    baselines,
    hold = false,
    surfaces,
  }: {
    adopt?: boolean;
    announce?: boolean;
    baselines: readonly StructuresById[];
    hold?: boolean;
    surfaces: SurfaceAvailability;
  }
): { account: AccountState; project: Project } => {
  let settledAccount = account;
  let next = project;

  if (project.legacyLayout) {
    const { legacyLayout, ...rest } = project;
    const { held: _held, heldView, ...stored } = legacyLayout;
    const migration = migrateProjectLayout(stored, {
      defaultWorkspaceId: defaultLayoutPreset.id,
      surfaces,
      workspaces: getAccountStructures(account),
    });
    const structures = Object.entries(migration.structures);
    const adopted = new Set(
      adopt ? structures.filter(([id, arrangement]) => isAdoptable(account, id, arrangement!)).map(([id]) => id) : []
    );
    const dropped = structures
      .filter(
        ([workspaceId, arrangement]) =>
          !adopted.has(workspaceId) &&
          baselines.every((baseline) => {
            const seen = baseline[workspaceId];

            return seen === undefined || !isSamePlacement(arrangement!, seen);
          })
      )
      .map(([workspaceId]) => workspaceId);
    const told = (heldView?.dropped ?? []).filter((id) => !heldView?.untold?.includes(id));
    const untold = dropped.filter((id) => !told.includes(id));

    for (const workspaceId of adopted) {
      settledAccount = writeWorkspaceStructure(settledAccount, workspaceId, migration.structures[workspaceId]!);
    }

    next = {
      ...rest,
      ...(announce && adopted.size > 0 ? { adoptedArrangements: [...adopted] } : {}),
      ...(untold.length > 0 ? { droppedArrangements: untold } : {}),
      ...(hold ? { heldLegacyLayout: { ...stored, held: { adopted: adopt, dropped } } } : {}),
      // An adopted arrangement is its workspace's structure now, so nothing is left to view differently.
      views: heldView?.views ?? Object.fromEntries(Object.entries(migration.views).filter(([id]) => !adopted.has(id))),
      workspaceId: heldView?.workspaceId ?? migration.workspaceId,
    };
  }

  const workspaceId = doesLayoutPresetExist(settledAccount, next.workspaceId)
    ? next.workspaceId
    : defaultLayoutPreset.id;
  const views = Object.keys(next.views).every((id) => doesLayoutPresetExist(settledAccount, id))
    ? next.views
    : Object.fromEntries(Object.entries(next.views).filter(([id]) => doesLayoutPresetExist(settledAccount, id)));
  const droppedArrangements = next.droppedArrangements?.filter((id) => doesLayoutPresetExist(settledAccount, id));
  const { droppedArrangements: _dropped, ...withoutNotice } = next;

  return {
    account: settledAccount,
    project:
      workspaceId === next.workspaceId &&
      views === next.views &&
      droppedArrangements?.length === next.droppedArrangements?.length
        ? next
        : {
            ...withoutNotice,
            ...(droppedArrangements && droppedArrangements.length > 0 ? { droppedArrangements } : {}),
            views,
            workspaceId,
          },
  };
};

/** Whether a project may adopt: one stored before the split by this account, not brought in by an import. */
const mayAdopt = (project: Project): boolean => project.legacyLayout !== undefined && !project.legacyLayout.imported;

/**
 * The project whose arrangements an account adopts: one that adopted in a load whose adoption was never saved, so the
 * choice holds across that reload; else the one opening as active; else the most recently stored.
 */
const findAdoptingProject = (projects: readonly Project[], activeProjectId: string): Project | undefined => {
  const stored = projects.filter(mayAdopt);
  const adoptedBefore = stored.filter((project) => project.legacyLayout!.heldView?.adopted);
  const candidates = adoptedBefore.length > 0 ? adoptedBefore : stored;

  return (
    candidates.find((project) => project.id === activeProjectId) ??
    candidates.reduce<Project | undefined>(
      (latest, project) =>
        latest === undefined || (project.legacyLayout!.storedAt ?? '') > (latest.legacyLayout!.storedAt ?? '')
          ? project
          : latest,
      undefined
    )
  );
};

/** `project` adopting into `account`, which marks itself adopted; held until a session write carrying it is saved. */
const adoptFrom = (account: AccountState, project: Project, surfaces: SurfaceAvailability, announce: boolean) => {
  const adopted = settleProjectLayout(account, project, {
    adopt: true,
    announce,
    baselines: [getAccountStructures(account), SHIPPED_STRUCTURES],
    hold: true,
    surfaces,
  });

  return { account: { ...adopted.account, workspaceLayoutsAdopted: true as const }, project: adopted.project };
};

/** Where settling left the account: `adoptionPending` while an adoption this editor made is not yet saved. */
export interface SettledLayouts {
  account: AccountState;
  adoptionPending: boolean;
}

/**
 * Hydration's layout step. An account that has not adopted (`workspaceLayoutsAdopted`) adopts, silently, the
 * arrangements of the project `findAdoptingProject` picks, and migrates every other project against the adopted
 * workspaces; each project it migrates holds its stored layout until a session write carrying that account is
 * acknowledged. With no project that may adopt, it adopts nothing and stays unadopted. An account that adopted only
 * migrates.
 */
export const settleHydratedProjects = (
  account: AccountState,
  projects: readonly Project[],
  activeProjectId: string,
  surfaces: SurfaceAvailability
): SettledLayouts & { projects: Project[] } => {
  const adopting =
    account.workspaceLayoutsAdopted === true ? undefined : findAdoptingProject(projects, activeProjectId);

  if (!adopting) {
    return {
      account,
      adoptionPending: false,
      projects: projects.map(
        (project) => settleProjectLayout(account, project, { baselines: [SHIPPED_STRUCTURES], surfaces }).project
      ),
    };
  }

  const adopted = adoptFrom(account, adopting, surfaces, false);
  const baselines = [getAccountStructures(account), SHIPPED_STRUCTURES];

  return {
    account: adopted.account,
    adoptionPending: true,
    projects: projects.map((project) =>
      project === adopting
        ? adopted.project
        : settleProjectLayout(adopted.account, project, { baselines, hold: true, surfaces }).project
    ),
  };
};

/**
 * Settles a project arriving after hydration. Into an account that has not adopted, a project that may adopt does, as
 * hydration's would, and is told which workspaces took its arrangements; otherwise its own arrangements are dropped,
 * and it is held while an adoption is pending, since it migrated against workspaces the server may not hold yet.
 */
export const settleArrivingProject = (
  account: AccountState,
  project: Project,
  { adoptionPending, surfaces }: { adoptionPending: boolean; surfaces: SurfaceAvailability }
): SettledLayouts & { project: Project } => {
  if (mayAdopt(project) && account.workspaceLayoutsAdopted !== true) {
    return { ...adoptFrom(account, project, surfaces, true), adoptionPending: true };
  }

  return {
    account,
    adoptionPending,
    project: settleProjectLayout(account, project, {
      baselines: [SHIPPED_STRUCTURES],
      hold: adoptionPending,
      surfaces,
    }).project,
  };
};

/** Once the account that adopted is saved, migrated projects stop writing their stored layout back. */
export const releaseHeldLayout = (project: Project): Project => {
  if (!project.heldLegacyLayout) {
    return project;
  }

  const { heldLegacyLayout: _held, ...released } = project;

  return released;
};
