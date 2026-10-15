import { useState } from 'react';

/** Bound kept mounts because each retains DOM and widgets may retain decoded images and measurement caches. */
export const MOUNTED_INSTANCE_LIMIT = 3;

/**
 * Remember recently shown instances independently of replaced region membership. Reset on project id because
 * instance ids repeat across projects. State is ephemeral; closing hides rather than destroys retained instances,
 * bounded by limit.
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
 * Exclude instances active in another region or floating window to prevent hidden duplicate mounts. Consult
 * active/floating ids, not preset-replaced membership lists.
 */
export const withoutInstancesShownElsewhere = (
  mountedIds: string[],
  activeId: string | undefined,
  activeIdsElsewhere: readonly string[]
): string[] => {
  const kept = mountedIds.filter((id) => id === activeId || !activeIdsElsewhere.includes(id));

  return kept.length === mountedIds.length ? mountedIds : kept;
};

/** The `activeInstanceId` of every region except `region`, plus every floating instance id. */
export const getActiveInstanceIdsOutside = (
  widgetRegions: Record<string, { activeInstanceId: string }>,
  region: string,
  floatingWidgets?: Record<string, unknown>
): string[] => [
  ...Object.entries(widgetRegions)
    .filter(([name]) => name !== region)
    .map(([, regionState]) => regionState.activeInstanceId),
  ...Object.keys(floatingWidgets ?? {}),
];

export const areInstanceIdListsEqual = (left: readonly string[], right: readonly string[]): boolean =>
  left.length === right.length && left.every((id, index) => id === right[index]);
