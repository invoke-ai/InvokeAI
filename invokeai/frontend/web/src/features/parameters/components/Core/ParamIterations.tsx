import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvNumberInput } from 'common/components/InvNumberInput/InvNumberInput';
import { setIterations } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';

const ParamIterations = () => {
  const iterations = useAppSelector((state) => state.generation.iterations);
  const dispatch = useAppDispatch();

  const onChange = useCallback(
    (v: number) => {
      dispatch(setIterations(v));
    },
    [dispatch]
  );

  return (
    <InvNumberInput
      step={1}
      min={1}
      max={999}
      onChange={onChange}
      value={iterations}
      h="full"
      w="216px"
      numberInputFieldProps={{
        ps: '144px',
        borderInlineStartRadius: 'base',
        h: 'full',
        textAlign: 'center',
      }}
    />
  );
};

export default memo(ParamIterations);
