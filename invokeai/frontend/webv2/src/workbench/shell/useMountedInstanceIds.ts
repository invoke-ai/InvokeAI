import { useState } from 'react';

/**
 * How many instances a region keeps mounted behind it. A kept canvas holds its
 * WebGL context and a kept gallery holds its virtualizer, so this is bounded
 * rather than unlimited; three covers every realistic back-and-forth.
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
 * Ephemeral by design. A remembered mount is a session affordance, not state
 * worth persisting.
 */
export const useMountedInstanceIds = (activeId: string | undefined, limit = MOUNTED_INSTANCE_LIMIT): string[] => {
  const [ids, setIds] = useState<string[]>(() => (activeId === undefined ? [] : [activeId]));
  const next =
    activeId === undefined || ids.at(-1) === activeId
      ? ids
      : [...ids.filter((id) => id !== activeId), activeId].slice(-limit);

  if (next !== ids) {
    setIds(next);
  }

  return next;
};
