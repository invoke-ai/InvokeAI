import { CompositeNumberInput } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectIterations, setIterations } from 'features/controlLayers/store/paramsSlice';
import { selectIterationsConfig } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';

export const QueueIterationsNumberInput = memo(() => {
  const iterations = useAppSelector(selectIterations);
  const config = useAppSelector(selectIterationsConfig);
  const dispatch = useAppDispatch();
  const handleChange = useCallback(
    (v: number) => {
      dispatch(setIterations(v));
    },
    [dispatch]
  );

  return (
    <InformationalPopover feature="paramIterations">
      <CompositeNumberInput
        step={config.coarseStep}
        fineStep={config.fineStep}
        min={1}
        max={999}
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
