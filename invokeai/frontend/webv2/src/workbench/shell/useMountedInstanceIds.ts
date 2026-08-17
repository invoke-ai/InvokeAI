import { useState } from 'react';

/**
 * How many instances a region keeps mounted behind it. A kept gallery holds its
 * virtualizer, its measurement cache and the decoded images in its window, and
 * every kept widget holds its DOM, so this is bounded rather than unlimited;
 * three covers every realistic back-and-forth.
 */
export const MOUNTED_INSTANCE_LIMIT = 3;

/**
 * The instance ids a region keeps mounted, most-recently-shown last.
 *
 * Deliberately independent of the region's own `instanceIds`: applying a preset
 * replaces those wholesale, so a widget the user is switching away from is no
 * longer in the region at all. Remembering it here is what lets `<Activity>`
 * hold it — and with it the canvas viewport, gallery scroll and selection that
 * a full unmount would take.
 *
 * `resetKey` is the active project id. Widget instance ids are deterministic and
 * identical in every project, so a remembered id does not go stale on a project
 * switch — it silently resolves to a real instance of the *new* project, and the
 * previous project's scroll offsets and in-widget selection would surface under
 * it. Dropping the set when the key changes is what keeps a session's memory
 * scoped to the project it belongs to.
 *
 * Ephemeral by design. A remembered mount is a session affordance, not state
 * worth persisting.
 *
 * Known and accepted: closing a widget reads as hiding it. `closeWidgetPlacement`
 * takes the instance out of its region but leaves it in `widgetInstances`, so a
 * closed widget is kept like any other and reopens with its state intact — the
 * way a desktop editor reopens a panel you closed. Telling a close apart from a
 * layout switch needs an explicit dismissal signal that does not exist yet; until
 * it does, `limit` is what bounds the consequence.
 */
export const useMountedInstanceIds = (
  activeId: string | undefined,
  resetKey: string,
  limit = MOUNTED_INSTANCE_LIMIT
): string[] => {
  const [remembered, setRemembered] = useState<{ ids: string[]; resetKey: string }>(() => ({
    ids: activeId === undefined ? [] : [activeId],
    resetKey,
  }));
  const ids = remembered.resetKey === resetKey ? remembered.ids : [];
  const next =
    activeId === undefined || ids.at(-1) === activeId
      ? ids
      : [...ids.filter((id) => id !== activeId), activeId].slice(-limit);

  if (next !== remembered.ids || resetKey !== remembered.resetKey) {
    setRemembered({ ids: next, resetKey });
  }

  return next;
};

/**
 * Kept ids, minus any instance another region is actively rendering.
 *
 * One instance can be placed in more than one region — `preview` is Compose's
 * centre view and also sits in Edit's and Automate's right rails — so a kept
 * hidden copy can end up shadowing a live one: the same instance mounted twice,
 * one of them a ghost the user can neither see nor dismiss.
 *
 * Only the regions' `activeInstanceId` values are consulted. Region membership
 * (`instanceIds`) is the thing a preset replaces wholesale, and deriving
 * anything here from it would reintroduce the very filtering bug this mechanism
 * exists to avoid. Actives are a small, stable signal that cannot.
 */
export const withoutInstancesShownElsewhere = (
  mountedIds: string[],
  activeId: string | undefined,
  activeIdsElsewhere: readonly string[]
): string[] => {
  const kept = mountedIds.filter((id) => id === activeId || !activeIdsElsewhere.includes(id));

  return kept.length === mountedIds.length ? mountedIds : kept;
};

/** The `activeInstanceId` of every region except `region`. */
export const getActiveInstanceIdsOutside = (
  widgetRegions: Record<string, { activeInstanceId: string }>,
  region: string
): string[] =>
  Object.entries(widgetRegions)
    .filter(([name]) => name !== region)
    .map(([, regionState]) => regionState.activeInstanceId);

export const areInstanceIdListsEqual = (left: readonly string[], right: readonly string[]): boolean =>
  left.length === right.length && left.every((id, index) => id === right[index]);
