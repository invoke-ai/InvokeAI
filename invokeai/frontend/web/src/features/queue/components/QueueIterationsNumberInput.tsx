import { CompositeNumberInput } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectIterations, setIterations, useParamsDispatch } from 'features/controlLayers/store/paramsSlice';
import { selectIterationsConfig } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';

export const QueueIterationsNumberInput = memo(() => {
  const iterations = useAppSelector(selectIterations);
  const config = useAppSelector(selectIterationsConfig);
  const dispatchParams = useParamsDispatch();
  const handleChange = useCallback(
    (v: number) => {
      dispatchParams(setIterations, v);
    },
    [dispatchParams]
  );

  return (
    <InformationalPopover feature="paramIterations">
      <CompositeNumberInput
        step={config.coarseStep}
        fineStep={config.fineStep}
        min={1}
        max={config.numberInputMax}
        onChange={handleChange}
        value={iterations}
        defaultValue={1}
        pos="absolute"
        insetInlineEnd={0}
        h="full"
        ps={0}
        w="72px"
        flexShrink={0}
        variant="iterations"
      />
    </InformationalPopover>
  );
});

QueueIterationsNumberInput.displayName = 'QueueIterationsNumberInput';
