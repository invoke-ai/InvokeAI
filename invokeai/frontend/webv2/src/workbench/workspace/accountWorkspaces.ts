import type {
  LayoutPreset,
  LayoutPresetId,
  LayoutPresetMetadataOverride,
  LayoutPresetRoute,
} from '@workbench/layoutContracts';
import type { AccountState } from '@workbench/projectContracts';

import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { getOrderedLayoutPresets } from '@workbench/layoutPresetCollection';
import { getLayoutPreset, isBuiltInLayoutPresetId, resolveLayoutPresetId } from '@workbench/layoutPresets';
import { resolveSavedLayoutPreset } from '@workbench/layoutPresetSnapshots';

import type { WorkspaceStructure } from './structure';

import { isSameStructure } from './structure';

/** A workspace's structure as the account saved it; the same object while the account does not change it. */
export const getWorkspaceStructure = (account: AccountState, workspaceId: LayoutPresetId): WorkspaceStructure =>
  resolveSavedLayoutPreset(account, workspaceId).snapshot;

/** Every workspace the account has, by id, as its saved structure. */
export const getAccountStructures = (account: AccountState): Partial<Record<LayoutPresetId, WorkspaceStructure>> =>
  Object.fromEntries(getOrderedLayoutPresets(account).map((preset) => [preset.id, preset.snapshot]));

/**
 * Everything the account holds for one workspace: a built-in's overrides (absent fields follow what it ships with),
 * or a custom workspace's own record (absent when the account has no such workspace).
 */
export type WorkspaceRecord =
  | {
      kind: 'built-in';
      metadata?: LayoutPresetMetadataOverride;
      route?: LayoutPresetRoute;
      structure?: WorkspaceStructure;
    }
  | { kind: 'custom'; preset?: LayoutPreset };

const withEntry = <Value>(
  entries: Partial<Record<LayoutPresetId, Value>> | undefined,
  workspaceId: LayoutPresetId,
  value: Value | undefined
): Partial<Record<LayoutPresetId, Value>> => {
  const { [workspaceId]: _previous, ...others } = entries ?? {};

  return value === undefined ? others : { ...others, [workspaceId]: value };
};

export const readWorkspaceRecord = (account: AccountState, workspaceId: LayoutPresetId): WorkspaceRecord =>
  isBuiltInLayoutPresetId(workspaceId)
    ? {
        kind: 'built-in',
        ...(account.layoutPresetMetadataOverrides?.[workspaceId]
          ? { metadata: account.layoutPresetMetadataOverrides[workspaceId] }
          : {}),
        ...(account.layoutPresetRouteOverrides?.[workspaceId]
          ? { route: account.layoutPresetRouteOverrides[workspaceId] }
          : {}),
        ...(account.layoutPresetOverrides?.[workspaceId]
          ? { structure: account.layoutPresetOverrides[workspaceId] }
          : {}),
      }
    : {
        kind: 'custom',
        ...(() => {
          const preset = account.customLayoutPresets?.find((candidate) => candidate.id === workspaceId);

          return preset ? { preset } : {};
        })(),
      };

/** The account noting that the workspace model wrote the workspace's structure, so adoption leaves it alone. */
const withWrittenStructure = (account: AccountState, workspaceId: LayoutPresetId): AccountState =>
  account.writtenStructureIds?.includes(workspaceId)
    ? account
    : { ...account, writtenStructureIds: [...(account.writtenStructureIds ?? []), workspaceId] };

export const isSameWorkspaceRecord = (left: WorkspaceRecord, right: WorkspaceRecord): boolean =>
  areJsonValuesStructurallyEqual(left, right);

/**
 * Puts a workspace's record back in the account: a built-in's overrides exactly as recorded, or a custom workspace's
 * record while the account still has it (a deleted workspace is never recreated). Nothing else changes.
 */
export const writeWorkspaceRecord = (
  account: AccountState,
  workspaceId: LayoutPresetId,
  record: WorkspaceRecord
): AccountState => {
  if (record.kind === 'built-in') {
    return withWrittenStructure(
      {
        ...account,
        layoutPresetMetadataOverrides: withEntry(account.layoutPresetMetadataOverrides, workspaceId, record.metadata),
        layoutPresetOverrides: withEntry(account.layoutPresetOverrides, workspaceId, record.structure),
        layoutPresetRouteOverrides: withEntry(account.layoutPresetRouteOverrides, workspaceId, record.route),
      },
      workspaceId
    );
  }

  const { preset } = record;

  return preset && account.customLayoutPresets?.some((candidate) => candidate.id === workspaceId)
    ? withWrittenStructure(
        {
          ...account,
          customLayoutPresets: account.customLayoutPresets.map((candidate) =>
            candidate.id === workspaceId ? preset : candidate
          ),
        },
        workspaceId
      )
    : account;
};

/**
 * An account with `structure` as the workspace's: a custom workspace's own snapshot, or a built-in's override, which
 * goes when the structure is the shipped one so the built-in follows what later releases ship. The same account when
 * nothing changes.
 */
export const writeWorkspaceStructure = (
  account: AccountState,
  workspaceId: LayoutPresetId,
  structure: WorkspaceStructure
): AccountState => {
  if (isSameStructure(getWorkspaceStructure(account, workspaceId), structure)) {
    return account;
  }

  const snapshot: WorkspaceStructure =
    structure.layout.presetId === workspaceId
      ? structure
      : { ...structure, layout: { ...structure.layout, presetId: workspaceId } };

  if (isBuiltInLayoutPresetId(workspaceId)) {
    const isShipped = isSameStructure(getLayoutPreset(workspaceId).snapshot, snapshot);

    return withWrittenStructure(
      {
        ...account,
        layoutPresetOverrides: withEntry(account.layoutPresetOverrides, workspaceId, isShipped ? undefined : snapshot),
      },
      workspaceId
    );
  }

  return withWrittenStructure(
    {
      ...account,
      customLayoutPresets: (account.customLayoutPresets ?? []).map((preset) =>
        preset.id === workspaceId ? { ...preset, snapshot } : preset
      ),
    },
    workspaceId
  );
};

/**
 * Renames a workspace. A built-in's label equal to its shipped one is no rename: storing it would freeze one
 * language. A blank label changes nothing.
 */
export const renameWorkspace = (account: AccountState, workspaceId: LayoutPresetId, label: string): AccountState => {
  const trimmed = label.trim();

  if (!trimmed) {
    return account;
  }

  if (isBuiltInLayoutPresetId(workspaceId)) {
    const { label: _label, ...rest } = account.layoutPresetMetadataOverrides?.[workspaceId] ?? {};
    const metadata = trimmed === getLayoutPreset(workspaceId).label ? rest : { ...rest, label: trimmed };

    return {
      ...account,
      layoutPresetMetadataOverrides: withEntry(
        account.layoutPresetMetadataOverrides,
        workspaceId,
        Object.keys(metadata).length > 0 ? metadata : undefined
      ),
    };
  }

  return {
    ...account,
    customLayoutPresets: (account.customLayoutPresets ?? []).map((preset) =>
      preset.id === workspaceId ? { ...preset, label: trimmed } : preset
    ),
  };
};

/** Sets a workspace's icon; a built-in's shipped icon is stored as none. */
export const setWorkspaceIcon = (account: AccountState, workspaceId: LayoutPresetId, iconId: string): AccountState => {
  if (isBuiltInLayoutPresetId(workspaceId)) {
    const { iconId: _iconId, ...rest } = account.layoutPresetMetadataOverrides?.[workspaceId] ?? {};
    const metadata = iconId === getLayoutPreset(workspaceId).iconId ? rest : { ...rest, iconId };

    return {
      ...account,
      layoutPresetMetadataOverrides: withEntry(
        account.layoutPresetMetadataOverrides,
        workspaceId,
        Object.keys(metadata).length > 0 ? metadata : undefined
      ),
    };
  }

  return {
    ...account,
    customLayoutPresets: (account.customLayoutPresets ?? []).map((preset) =>
      preset.id === workspaceId ? { ...preset, iconId } : preset
    ),
  };
};

const isSameRoute = (left: LayoutPresetRoute, right: LayoutPresetRoute | undefined) =>
  right !== undefined && left.sourceId === right.sourceId && left.destination === right.destination;

/**
 * Sets a workspace's explicit route, `null` for Auto. A built-in always has its shipped route to fall back on, so Auto
 * or the shipped route removes its override.
 */
export const setWorkspaceRoute = (
  account: AccountState,
  workspaceId: LayoutPresetId,
  defaultRoute: LayoutPresetRoute | null
): AccountState => {
  if (isBuiltInLayoutPresetId(workspaceId)) {
    const isShipped = defaultRoute === null || isSameRoute(defaultRoute, getLayoutPreset(workspaceId).defaultRoute);

    return {
      ...account,
      layoutPresetRouteOverrides: withEntry(
        account.layoutPresetRouteOverrides,
        workspaceId,
        isShipped ? undefined : { ...defaultRoute }
      ),
    };
  }

  return {
    ...account,
    customLayoutPresets: (account.customLayoutPresets ?? []).map((preset) => {
      if (preset.id !== workspaceId) {
        return preset;
      }

      const { defaultRoute: _previous, ...withoutRoute } = preset;

      return defaultRoute ? { ...withoutRoute, defaultRoute: { ...defaultRoute } } : withoutRoute;
    }),
  };
};

/** A built-in as it ships: its structure, name, icon and route overrides removed. */
export const restoreWorkspaceDefault = (account: AccountState, workspaceId: LayoutPresetId): AccountState =>
  isBuiltInLayoutPresetId(workspaceId) ? writeWorkspaceRecord(account, workspaceId, { kind: 'built-in' }) : account;

/** An id a new workspace may take: a custom one the account does not have yet. */
export const isNewWorkspaceId = (account: AccountState, workspaceId: LayoutPresetId): boolean =>
  workspaceId.trim().length > 0 &&
  !isBuiltInLayoutPresetId(resolveLayoutPresetId(workspaceId)) &&
  !getOrderedLayoutPresets(account).some((preset) => preset.id === workspaceId);

/** The account with a new custom workspace, last in the tab order and active. Nothing else changes. */
export const createWorkspace = (account: AccountState, preset: LayoutPreset): AccountState =>
  withWrittenStructure(
    {
      ...account,
      activeLayoutPresetId: preset.id,
      customLayoutPresets: [...(account.customLayoutPresets ?? []), preset],
      layoutPresetOrder: [...getOrderedLayoutPresets(account).map((candidate) => candidate.id), preset.id],
    },
    preset.id
  );
