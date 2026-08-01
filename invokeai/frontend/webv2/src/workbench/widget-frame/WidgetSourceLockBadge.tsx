import type { WidgetTypeId } from '@workbench/widgetContracts';

import { Icon } from '@chakra-ui/react';
import { Tooltip } from '@platform/ui/Tooltip';
import { getSourceIdForWidgetTypeId } from '@workbench/graphWidgets';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { LockKeyholeIcon } from 'lucide-react';

/**
 * Marks the widget the invoke source is pinned to.
 *
 * Only the widget itself carries this. The preset strip does not: presets are
 * arrangements, not widgets, and marking one would put back exactly the
 * conflation between "which layout am I in" and "what will run" that the
 * routing model exists to remove.
 */
export const WidgetSourceLockBadge = ({ typeId }: { typeId?: WidgetTypeId }) => {
  const sourceId = typeId ? getSourceIdForWidgetTypeId(typeId) : null;
  const isLockedSource = useActiveProjectSelector(
    (project) => project.invocation.sourceLocked && project.invocation.sourceId === sourceId
  );

  if (!isLockedSource) {
    return null;
  }

  return (
    <Tooltip content="Invoke is locked to this widget. Unlock it from the routing menu." showArrow>
      <Icon as={LockKeyholeIcon} aria-label="Locked invoke source" boxSize="3" color="accent.fg" flexShrink={0} />
    </Tooltip>
  );
};
