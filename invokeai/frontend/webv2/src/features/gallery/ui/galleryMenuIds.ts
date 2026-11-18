import { useId, useMemo } from 'react';

/** Share the trigger ID between menu and Tooltip to preserve anchoring. */
export const useMenuTriggerIds = (): { trigger: string } => {
  const triggerId = useId();

  return useMemo(() => ({ trigger: triggerId }), [triggerId]);
};
