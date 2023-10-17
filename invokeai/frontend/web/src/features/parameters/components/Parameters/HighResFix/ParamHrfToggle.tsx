import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setHrfEnabled } from 'features/parameters/store/generationSlice';
import { ChangeEvent, useCallback } from 'react';

export default function ParamHrfToggle() {
  const dispatch = useAppDispatch();
  const tooltip =
    'Generate at an Initial Resolution then run Image to Image at the Base Resolution.';

  const hrfEnabled = useAppSelector(
    (state: RootState) => state.generation.hrfEnabled
  );

  const handleHrfEnabled = useCallback(
    (e: ChangeEvent<HTMLInputElement>) =>
      dispatch(setHrfEnabled(e.target.checked)),
    [dispatch]
  );

  return (
    <IAISwitch
      label="Enable High Resolution Fix"
      isChecked={hrfEnabled}
      onChange={handleHrfEnabled}
      tooltip={tooltip}
    />
  );
}
