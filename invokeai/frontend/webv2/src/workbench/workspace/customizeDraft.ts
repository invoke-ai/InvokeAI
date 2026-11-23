import type { LayoutPreset, LayoutPresetId } from '@workbench/layoutContracts';
import type { AccountState } from '@workbench/projectContracts';

import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { getLayoutPreset, isBuiltInLayoutPresetId } from '@workbench/layoutPresets';
import { resolveSavedLayoutPreset } from '@workbench/layoutPresetSnapshots';

import type { WorkspaceStructure } from './structure';
import type {
  CustomizeDraft,
  CustomizeHistoryEntry,
  CustomizeProjectSide,
  CustomizeSession,
  NewWorkspaceDraft,
} from './structureLock';

import {
  createWorkspace,
  renameWorkspace,
  restoreWorkspaceDefault,
  setWorkspaceIcon,
  setWorkspaceRoute,
  writeWorkspaceStructure,
} from './accountWorkspaces';
import { normalizeStructure } from './structure';

/** Steps a layout history keeps, a Customize session's and every project's. */
export const LAYOUT_HISTORY_LIMIT = 40;
/** Layout edits sharing a merge key within this window fold into one step. */
export const LAYOUT_MERGE_WINDOW_MS = 1500;

/**
 * What a session's absent draft fields read as: the built-in as it ships after Reset to default, else the workspace
 * as the account saved it. A new workspace's draft names every field itself, so it has none.
 */
export const getCustomizeBase = (account: AccountState, session: CustomizeSession): LayoutPreset | null =>
  session.newWorkspaceId !== undefined
    ? null
    : session.resetPending && isBuiltInLayoutPresetId(session.presetId)
      ? getLayoutPreset(session.presetId)
      : resolveSavedLayoutPreset(account, session.presetId);

const toNewWorkspaceDraft = ({ defaultRoute, iconId, label }: NewWorkspaceDraft): CustomizeDraft => ({
  defaultRoute,
  ...(iconId === undefined ? {} : { iconId }),
  label,
});

/** A session on `structure`, the account's for an existing workspace or a new workspace's starting copy. */
export const beginCustomizeSession = ({
  entry,
  newWorkspace,
  presetId,
  structure,
}: {
  /** The project's view and route as Customize begins. */
  entry: CustomizeProjectSide;
  newWorkspace?: NewWorkspaceDraft;
  presetId: LayoutPresetId;
  structure: WorkspaceStructure;
}): CustomizeSession => ({
  draft: newWorkspace ? toNewWorkspaceDraft(newWorkspace) : {},
  entryInvocation: entry.invocation,
  entryStructure: structure,
  entryView: entry.view,
  history: { future: [], past: [] },
  ...(newWorkspace ? { newWorkspaceId: newWorkspace.id } : {}),
  presetId,
  structure,
});

/** Reset to default: the built-in's shipped structure and metadata; the caller empties the view. Null when not one. */
export const resetCustomizeSession = (session: CustomizeSession): CustomizeSession | null =>
  session.newWorkspaceId !== undefined || !isBuiltInLayoutPresetId(session.presetId)
    ? null
    : {
        ...session,
        draft: {},
        resetPending: true,
        structure: normalizeStructure(getLayoutPreset(session.presetId).snapshot),
      };

/** The session with name, icon or route edits merged in; the same session when they change nothing. */
export const editCustomizeDraft = (session: CustomizeSession, draft: CustomizeDraft): CustomizeSession =>
  Object.entries(draft).every(([field, value]) => session.draft[field as keyof CustomizeDraft] === value)
    ? session
    : { ...session, draft: { ...session.draft, ...draft } };

/**
 * Save as new: the session goes on as a new workspace's draft. Earlier steps keep their arrangement; the copy's name,
 * icon and route are its own from here on. Null for a session already drafting one.
 */
export const saveCustomizeSessionAsNew = (
  session: CustomizeSession,
  workspace: NewWorkspaceDraft
): CustomizeSession | null => {
  if (session.newWorkspaceId !== undefined) {
    return null;
  }

  const draft = toNewWorkspaceDraft(workspace);
  const carry = ({ resetPending: _reset, ...entry }: CustomizeHistoryEntry): CustomizeHistoryEntry => ({
    ...entry,
    draft,
  });
  const { resetPending: _resetPending, ...rest } = session;

  return {
    ...rest,
    draft,
    history: { future: session.history.future.map(carry), past: session.history.past.map(carry) },
    newWorkspaceId: workspace.id,
  };
};

const toHistoryEntry = (
  session: CustomizeSession,
  { invocation, view }: CustomizeProjectSide
): Omit<CustomizeHistoryEntry, 'mergedAt' | 'mergeKey'> => ({
  draft: session.draft,
  invocation,
  ...(session.resetPending ? { resetPending: true as const } : {}),
  structure: session.structure,
  view,
});

/** Whether a step changed the session's draft, its metadata, the reset, the view it is seen through or the route. */
export const isCustomizeStep = (
  before: CustomizeSession,
  after: CustomizeSession,
  sideBefore: CustomizeProjectSide,
  sideAfter: CustomizeProjectSide
): boolean =>
  after.structure !== before.structure ||
  after.draft !== before.draft ||
  after.resetPending !== before.resetPending ||
  sideAfter.view !== sideBefore.view ||
  sideAfter.invocation !== sideBefore.invocation;

/** Records the state before a step; a step with the previous one's merge key, within the window, folds into it. */
export const pushCustomizeHistory = (
  session: CustomizeSession,
  before: CustomizeSession,
  sideBefore: CustomizeProjectSide,
  mergeKey: string | undefined,
  timestamp: string
): CustomizeSession => {
  const { future, past } = session.history;
  const last = past.at(-1);

  if (
    mergeKey &&
    last?.mergeKey === mergeKey &&
    future.length === 0 &&
    Date.parse(timestamp) - Date.parse(last.mergedAt) <= LAYOUT_MERGE_WINDOW_MS
  ) {
    return { ...session, history: { future: [], past: [...past.slice(0, -1), { ...last, mergedAt: timestamp }] } };
  }

  return {
    ...session,
    history: {
      future: [],
      past: [
        ...past,
        { ...toHistoryEntry(before, sideBefore), ...(mergeKey ? { mergeKey } : {}), mergedAt: timestamp },
      ].slice(-LAYOUT_HISTORY_LIMIT),
    },
  };
};

/** Undo or Redo inside Customize: the draft, the view it is seen through and the route step together. Null at either end. */
export const stepCustomizeHistory = (
  session: CustomizeSession,
  side: CustomizeProjectSide,
  direction: 'undo' | 'redo',
  timestamp: string
): ({ session: CustomizeSession } & CustomizeProjectSide) | null => {
  const { future, past } = session.history;
  const entry = direction === 'undo' ? past.at(-1) : future[0];

  if (!entry) {
    return null;
  }

  const current: CustomizeHistoryEntry = { ...toHistoryEntry(session, side), mergedAt: timestamp };
  const { resetPending: _resetPending, ...rest } = session;

  return {
    session: {
      ...rest,
      draft: entry.draft,
      ...(entry.resetPending ? { resetPending: true } : {}),
      history:
        direction === 'undo'
          ? { future: [current, ...future].slice(0, LAYOUT_HISTORY_LIMIT), past: past.slice(0, -1) }
          : { future: future.slice(1), past: [...past, current].slice(-LAYOUT_HISTORY_LIMIT) },
      structure: entry.structure,
    },
    invocation: entry.invocation,
    view: entry.view,
  };
};

/**
 * Done, from the session alone, given the structure the screen shows as its defaults (`captured`) and whether the view
 * changed during the session. A new workspace is created (last in the tab order, active); an existing one gets its
 * overrides removed first when a reset is pending, its structure only when the session touched it (so an untouched
 * built-in is never pinned to this release's arrangement), and each name, icon or route the draft names. Only that
 * workspace's entries change. Null when the draft's name is blank.
 */
export const finishCustomizeSession = (
  account: AccountState,
  session: CustomizeSession,
  captured: WorkspaceStructure,
  viewChanged: boolean
): AccountState | null => {
  const { draft } = session;

  if (draft.label !== undefined && !draft.label.trim()) {
    return null;
  }

  if (session.newWorkspaceId !== undefined) {
    const id = session.newWorkspaceId;

    return createWorkspace(account, {
      ...(draft.defaultRoute ? { defaultRoute: { ...draft.defaultRoute } } : {}),
      ...(draft.iconId === undefined ? {} : { iconId: draft.iconId }),
      id,
      label: draft.label?.trim() ?? id,
      snapshot: { ...captured, layout: { ...captured.layout, presetId: id } },
    });
  }

  const id = session.presetId;
  const isTouched = session.structure !== session.entryStructure || session.resetPending === true || viewChanged;
  let next = session.resetPending ? restoreWorkspaceDefault(account, id) : account;

  if (isTouched) {
    next = writeWorkspaceStructure(next, id, captured);
  }
  if (draft.label !== undefined) {
    next = renameWorkspace(next, id, draft.label);
  }
  if (draft.iconId !== undefined) {
    next = setWorkspaceIcon(next, id, draft.iconId);
  }
  if (draft.defaultRoute !== undefined) {
    next = setWorkspaceRoute(next, id, draft.defaultRoute);
  }

  return areJsonValuesStructurallyEqual(next, account) ? account : next;
};
