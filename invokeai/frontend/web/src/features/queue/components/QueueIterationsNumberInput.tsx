import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import { InvNumberInput } from 'common/components/InvNumberInput/InvNumberInput';
import type { InvNumberInputFieldProps } from 'common/components/InvNumberInput/types';
import { setIterations } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';

const numberInputFieldProps: InvNumberInputFieldProps = {
  ps: 6,
  borderInlineStartRadius: 'base',
  h: 'full',
  textAlign: 'center',
  fontSize: 'md',
  fontWeight: 'semibold',
};

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
    <IAIInformationalPopover feature="paramIterations">
      <InvNumberInput
        step={coarseStep}
        fineStep={fineStep}
        min={1}
        max={999}
        onChange={handleChange}
        value={iterations}
        defaultValue={1}
        numberInputFieldProps={numberInputFieldProps}
        pos="absolute"
        insetInlineEnd={0}
        h="full"
        ps={0}
        w="72px"
        flexShrink={0}
        variant="darkFilled"
      />
    </IAIInformationalPopover>
  );
});

QueueIterationsNumberInput.displayName = 'QueueIterationsNumberInput';
