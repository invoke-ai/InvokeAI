import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setHrfToggle } from 'features/parameters/store/generationSlice';
import { ChangeEvent } from 'react';

export default function ParamHrfToggle() {
  const dispatch = useAppDispatch();

  const hrfToggled = useAppSelector(
    (state: RootState) => state.generation.hrfToggled
  );

  const handleHrfToggle = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setHrfToggle(e.target.checked));

  return (
    <IAISwitch
      label="Toggle High Resolution Fix"
      isChecked={hrfToggled}
      onChange={handleHrfToggle}
    />
  );
}
