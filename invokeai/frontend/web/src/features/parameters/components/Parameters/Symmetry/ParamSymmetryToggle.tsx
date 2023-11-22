import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setShouldUseSymmetry } from 'features/parameters/store/generationSlice';
import { ChangeEvent, useCallback } from 'react';

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
    <IAISwitch
      label="Enable Symmetry"
      isChecked={shouldUseSymmetry}
      onChange={handleChange}
    />
  );
}
