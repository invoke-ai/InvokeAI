import { NumberInput } from '@chakra-ui/react';
import { MIN_BATCH_COUNT, sanitizeBatchCount } from '@workbench/generation/batch';
import { useWidgetValuesSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';

const getBatchCount = (values: Record<string, unknown>): number => {
  const batchCount = values.batchCount;

  return sanitizeBatchCount(batchCount);
};

export const BatchCountField = () => {
  const batchCount = useWidgetValuesSelector('generate', getBatchCount);
  const dispatch = useWorkbenchDispatch();
  const handleValueChange = useCallback(
    ({ valueAsNumber }: { valueAsNumber: number }) => {
      if (Number.isFinite(valueAsNumber)) {
        dispatch({ batchCount: valueAsNumber, type: 'setGenerateBatchCount' });
      }
    },
    [dispatch]
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
