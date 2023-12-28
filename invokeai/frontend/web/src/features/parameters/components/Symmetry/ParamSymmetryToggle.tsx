import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSwitch } from 'common/components/InvSwitch/wrapper';
import { setShouldUseSymmetry } from 'features/parameters/store/generationSlice';
import type { ChangeEvent } from 'react';
import { useCallback } from 'react';

export default function ParamSymmetryToggle() {
  const shouldUseSymmetry = useAppSelector(
    (state: RootState) => state.generation.shouldUseSymmetry
  );

  const dispatch = useAppDispatch();
  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setShouldUseSymmetry(e.target.checked));
    },
    [dispatch]
  );

  return (
    <InvControl label="Enable Symmetry">
      <InvSwitch isChecked={shouldUseSymmetry} onChange={handleChange} />
    </InvControl>
  );
}
