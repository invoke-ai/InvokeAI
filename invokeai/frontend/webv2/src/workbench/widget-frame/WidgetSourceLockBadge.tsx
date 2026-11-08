import type { WidgetTypeId } from '@workbench/widgetContracts';

import { Icon } from '@chakra-ui/react';
import { Tooltip } from '@platform/ui/Tooltip';
import { getSourceIdForWidgetTypeId } from '@workbench/graphWidgets';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { LockKeyholeIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

/** Mark the pinned source widget only; presets describe arrangement rather than invocation ownership. */
export const WidgetSourceLockBadge = ({ typeId }: { typeId?: WidgetTypeId }) => {
  const { t } = useTranslation();
  const sourceId = typeId ? getSourceIdForWidgetTypeId(typeId) : null;
  const isLockedSource = useActiveProjectSelector(
    (project) => project.invocation.sourceLocked && project.invocation.sourceId === sourceId
  );

  if (!isLockedSource) {
    return null;
  }

  return (
    <Tooltip content={t('widgets.sourceLock.tooltip')} showArrow>
      <Icon
        as={LockKeyholeIcon}
        aria-label={t('widgets.sourceLock.label')}
        boxSize="3"
        color="accent.fg"
        flexShrink={0}
      />
    </Tooltip>
  );
};
