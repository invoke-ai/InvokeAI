import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setHrfEnabled } from 'features/parameters/store/generationSlice';
import { ChangeEvent } from 'react';

export default function ParamHrfToggle() {
  const dispatch = useAppDispatch();

  const hrfEnabled = useAppSelector(
    (state: RootState) => state.generation.hrfEnabled
  );

  const handleHrfEnabled = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setHrfEnabled(e.target.checked));

  return (
    <IAISwitch
      label="Toggle High Resolution Fix"
      isChecked={hrfEnabled}
      onChange={handleHrfEnabled}
    />
  );
}
