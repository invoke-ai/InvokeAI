import { NumberInput } from '@chakra-ui/react';
import { MIN_BATCH_COUNT, sanitizeBatchCount } from '@features/generation/settings';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';

const getBatchCount = (values: Record<string, unknown>): number => {
  const batchCount = values.batchCount;

  return sanitizeBatchCount(batchCount);
};

export const BatchCountField = () => {
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
      if (Number.isFinite(valueAsNumber)) {
        if (sourceId === 'upscale') {
          widgets.patchValues('upscale', { batchCount: valueAsNumber });
        } else {
          generation.setBatchCount(valueAsNumber);
        }
      }
    },
    [generation, sourceId, widgets]
  );

  return (
    <NumberInput.Root
      allowMouseWheel
      flexShrink={0}
      min={MIN_BATCH_COUNT}
      size="sm"
      value={String(batchCount)}
      w="14"
      onValueChange={handleValueChange}
    >
      <NumberInput.Control />
      <NumberInput.Input paddingStart="2.5" aria-label="Batch count" />
    </NumberInput.Root>
  );
};
