import { CompositeNumberInput } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { setIterations } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';

export const QueueIterationsNumberInput = memo(() => {
  const iterations = useAppSelector((s) => s.generation.iterations);
  const coarseStep = useAppSelector((s) => s.config.sd.iterations.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.iterations.fineStep);
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
        step={coarseStep}
        fineStep={fineStep}
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
