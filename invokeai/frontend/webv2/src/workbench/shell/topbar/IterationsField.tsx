import { NumberInput } from '@chakra-ui/react';
import { MIN_BATCH_COUNT } from '@features/generation/settings';
import { Tooltip } from '@platform/ui/Tooltip';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { getBatchCount } from './useInvocationState';

export const IterationsField = () => {
  const { t } = useTranslation();
  const { batchCount, sourceId } = useActiveProjectSelector(
    (project) => {
      const sourceId = project.invocation.sourceId;
      const typeId = sourceId === 'upscale' || sourceId === 'video' || sourceId === 'workflow' ? sourceId : 'generate';

      // The same resolution the patch command writes through, so reader and writer agree on the instance.
      return { batchCount: getBatchCount(getProjectWidgetValues(project, typeId)), sourceId };
    },
    (left, right) => left.batchCount === right.batchCount && left.sourceId === right.sourceId
  );
  const { generation, widgets } = useWorkbenchCommands();
  const handleValueChange = useCallback(
    ({ valueAsNumber }: { valueAsNumber: number }) => {
      if (!Number.isFinite(valueAsNumber)) {
        return;
      }

      if (sourceId === 'upscale' || sourceId === 'video' || sourceId === 'workflow') {
        widgets.patchValues(sourceId, { batchCount: valueAsNumber });
      } else {
        generation.setBatchCount(valueAsNumber);
      }
    },
    [generation, sourceId, widgets]
  );

  return (
    <Tooltip content={t('topbar.iterations.tooltip', { count: batchCount })} showArrow>
      <NumberInput.Root
        allowMouseWheel
        flexShrink={0}
        min={MIN_BATCH_COUNT}
        rounded="control"
        size="sm"
        value={String(batchCount)}
        w="14"
        onValueChange={handleValueChange}
      >
        <NumberInput.Control />
        {/* The visible surface is the inner control, so it inherits the root's
            radius — that is the corner an attached `Group` adjusts. */}
        <NumberInput.Input aria-label={t('topbar.iterations.label')} paddingStart="2" rounded="inherit" />
      </NumberInput.Root>
    </Tooltip>
  );
};
