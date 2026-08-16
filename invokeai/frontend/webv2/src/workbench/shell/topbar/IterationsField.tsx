import { NumberInput } from '@chakra-ui/react';
import { MIN_BATCH_COUNT } from '@features/generation/settings';
import { Tooltip } from '@platform/ui/Tooltip';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { getBatchCount } from './useInvocationState';

/**
 * How many times the current graph runs per Invoke.
 *
 * A plain `NumberInput` with its stepper — typing, arrow keys, and the wheel all
 * behave the way they do everywhere else in the app, because it is the same
 * control everywhere else in the app uses.
 */
export const IterationsField = () => {
  const { t } = useTranslation();
  const { batchCount, sourceId } = useActiveProjectSelector(
    (project) => {
      const sourceId = project.invocation.sourceId;
      const typeId = sourceId === 'upscale' ? 'upscale' : 'generate';
      const instance = Object.values(project.widgetInstances).find((candidate) => candidate.typeId === typeId);

      return { batchCount: getBatchCount(instance?.state.values ?? {}), sourceId };
    },
    (left, right) => left.batchCount === right.batchCount && left.sourceId === right.sourceId
  );
  const { generation, widgets } = useWorkbenchCommands();
  const handleValueChange = useCallback(
    ({ valueAsNumber }: { valueAsNumber: number }) => {
      if (!Number.isFinite(valueAsNumber)) {
        return;
      }

      if (sourceId === 'upscale') {
        widgets.patchValues('upscale', { batchCount: valueAsNumber });
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
        rounded="l2"
        size="xs"
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
