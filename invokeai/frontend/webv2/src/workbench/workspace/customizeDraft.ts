import type { RouteMove, RouteTarget } from '@workbench/invocationContracts';
import type { LayoutPreset, LayoutPresetId } from '@workbench/layoutContracts';
import type { AccountState } from '@workbench/projectContracts';

import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { getLayoutPreset, isBuiltInLayoutPresetId } from '@workbench/layoutPresets';
import { resolveSavedLayoutPreset } from '@workbench/layoutPresetSnapshots';

import type { WorkspaceStructure } from './structure';
import type { CustomizeDraft, CustomizeHistoryEntry, CustomizeSession, NewWorkspaceDraft } from './structureLock';
import type { WorkspaceView } from './view';

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

const toRouteTarget = ({ destination, sourceId }: RouteTarget): RouteTarget => ({ destination, sourceId });

const isAtRoute = (route: RouteTarget, target: RouteTarget): boolean =>
  route.sourceId === target.sourceId && route.destination === target.destination;

/** The move a layout action's auto-routing made, from the route before it to the route after; none when it made none. */
export const getRouteMove = (before: RouteTarget, after: RouteTarget): RouteMove | undefined =>
  isAtRoute(before, after) ? undefined : { from: toRouteTarget(before), to: toRouteTarget(after) };

const reverseRouteMove = (move: RouteMove | undefined): RouteMove | undefined =>
  move && { from: move.to, to: move.from };

/** The route with `move` stepped: only while it is still where the move starts, and never its locks. Null otherwise. */
export const stepRouteMove = <Route extends RouteTarget>(route: Route, move: RouteMove | undefined): Route | null =>
  move && isAtRoute(route, move.from) && !isAtRoute(route, move.to) ? { ...route, ...move.to } : null;

/**
 * Where taking back every step still in the session's history puts the route, as a move from where it is now: each
 * step's auto-routing move, newest first, the way Undo takes it, skipping one the route has since left. For Cancel and
 * for undoing Done. None when nothing would move it.
 */
export const getCustomizeRouteReturn = (session: CustomizeSession, route: RouteTarget): RouteMove | undefined =>
  getRouteMove(
    route,
    session.history.past.reduceRight<RouteTarget>(
      (current, entry) => stepRouteMove(current, entry.route) ?? current,
      route
    )
  );

/** A session on `structure`, the account's for an existing workspace or a new workspace's starting copy. */
export const beginCustomizeSession = ({
  entryView,
  newWorkspace,
  presetId,
  structure,
}: {
  entryView: WorkspaceView | undefined;
  newWorkspace?: NewWorkspaceDraft;
  presetId: LayoutPresetId;
  structure: WorkspaceStructure;
}): CustomizeSession => ({
  draft: newWorkspace ? toNewWorkspaceDraft(newWorkspace) : {},
  entryStructure: structure,
  entryView,
  history: { future: [], past: [] },
  ...(newWorkspace ? { newWorkspaceId: newWorkspace.id } : {}),
  presetId,
  structure,
});

/**
 * Whether ending the session now discards something: an arrangement other than the one it started from, a pending
 * reset, or a name, icon or route edit to an existing workspace. A new workspace's draft starts as a copy, so only its
 * arrangement counts.
 */
export const hasCustomizeChanges = (session: CustomizeSession): boolean =>
  session.resetPending === true ||
  (session.newWorkspaceId === undefined && Object.keys(session.draft).length > 0) ||
  !areJsonValuesStructurallyEqual(session.structure, session.entryStructure);

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
  view: WorkspaceView | undefined,
  route: RouteMove | undefined
): Omit<CustomizeHistoryEntry, 'mergedAt' | 'mergeKey'> => ({
  draft: session.draft,
  ...(session.resetPending ? { resetPending: true as const } : {}),
  ...(route ? { route } : {}),
  structure: session.structure,
  view,
});

/** Whether a step changed the session's draft, its metadata, the reset, the view it is seen through or the route. */
export const isCustomizeStep = (
  before: CustomizeSession,
  after: CustomizeSession,
  viewBefore: WorkspaceView | undefined,
  viewAfter: WorkspaceView | undefined,
  route: RouteMove | undefined
): boolean =>
  after.structure !== before.structure ||
  after.draft !== before.draft ||
  after.resetPending !== before.resetPending ||
  viewAfter !== viewBefore ||
  route !== undefined;

/**
 * Records the state before a step and the move its auto-routing made (`route`); a step with the previous one's merge
 * key, within the window, folds into it.
 */
export const pushCustomizeHistory = (
  session: CustomizeSession,
  before: CustomizeSession,
  viewBefore: WorkspaceView | undefined,
  route: RouteMove | undefined,
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
    const back = route ? { from: route.to, to: last.route?.to ?? route.from } : last.route;

    return {
      ...session,
      history: {
        future: [],
        past: [...past.slice(0, -1), { ...last, ...(back ? { route: back } : {}), mergedAt: timestamp }],
      },
    };
  }

  return {
    ...session,
    history: {
      future: [],
      past: [
        ...past,
        {
          ...toHistoryEntry(before, viewBefore, reverseRouteMove(route)),
          ...(mergeKey ? { mergeKey } : {}),
          mergedAt: timestamp,
        },
      ].slice(-LAYOUT_HISTORY_LIMIT),
    },
  };
};

/**
 * Undo or Redo inside Customize: the draft and the view it is seen through step together, and the route steps back or
 * forward with them only where the step's auto-routing moved it and nothing has picked another route since. Null at
 * either end.
 */
export const stepCustomizeHistory = <Route extends RouteTarget>(
  session: CustomizeSession,
  view: WorkspaceView | undefined,
  route: Route,
  direction: 'undo' | 'redo',
  timestamp: string
): { invocation: Route; session: CustomizeSession; view: WorkspaceView | undefined } | null => {
  const { future, past } = session.history;
  const entry = direction === 'undo' ? past.at(-1) : future[0];

  if (!entry) {
    return null;
  }

  const stepped = stepRouteMove(route, entry.route);
  const current: CustomizeHistoryEntry = {
    ...toHistoryEntry(session, view, reverseRouteMove(entry.route)),
    mergedAt: timestamp,
  };
  const { resetPending: _resetPending, ...rest } = session;

  return {
    invocation: stepped ?? route,
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
