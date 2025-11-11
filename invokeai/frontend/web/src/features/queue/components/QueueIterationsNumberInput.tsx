import { CompositeNumberInput } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectIterations, setIterations } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';

export const QueueIterationsNumberInput = memo(() => {
  const dispatch = useAppDispatch();
  const iterations = useAppSelector(selectIterations);

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setIterations(v));
    },
    [dispatch]
  );

  return (
    <InformationalPopover feature="paramIterations">
      <CompositeNumberInput
        step={1}
        fineStep={1}
        min={1}
        max={10000}
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
