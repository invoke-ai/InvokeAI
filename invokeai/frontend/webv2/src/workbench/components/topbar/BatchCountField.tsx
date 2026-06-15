import { NumberInput } from '@chakra-ui/react';
import { MIN_BATCH_COUNT, sanitizeBatchCount } from '@workbench/generation/batch';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';

const getBatchCount = (values: Record<string, unknown>): number => {
  const batchCount = values.batchCount;

  return sanitizeBatchCount(batchCount);
};

export const BatchCountField = () => {
  const generateValues = useActiveProjectSelector((project) => project.widgetStates.generate.values);
  const dispatch = useWorkbenchDispatch();
  const batchCount = getBatchCount(generateValues);

  return (
    <NumberInput.Root
      allowMouseWheel
      flexShrink={0}
      min={MIN_BATCH_COUNT}
      size="sm"
      value={String(batchCount)}
      w="14"
      onValueChange={({ valueAsNumber }) => {
        if (Number.isFinite(valueAsNumber)) {
          dispatch({ batchCount: valueAsNumber, type: 'setGenerateBatchCount' });
        }
      }}
    >
      <NumberInput.Control />
      <NumberInput.Input paddingStart="2.5" aria-label="Batch count" />
    </NumberInput.Root>
  );
};
