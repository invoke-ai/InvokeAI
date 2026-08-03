import { useId, useMemo } from 'react';

/**
 * Shared trigger id for a menu whose trigger a `Tooltip` wraps. Both machines
 * render onto one element and each wants to own its id; without this the menu
 * has no anchor and opens at the viewport origin.
 *
 * @see workbench/shell/topbar/RoutingControl.tsx
 */
export const useMenuTriggerIds = (): { trigger: string } => {
  const triggerId = useId();

  return useMemo(() => ({ trigger: triggerId }), [triggerId]);
};
