import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setHrfEnabled } from 'features/parameters/store/generationSlice';
import { ChangeEvent, useCallback, useEffect } from 'react';

export default function ParamHrfToggle() {
  const dispatch = useAppDispatch();
  const tooltip =
    'Generate with a lower initial resolution, upscale to base resolution, process run Image-to-Image.';

  const hrfEnabled = useAppSelector(
    (state: RootState) => state.generation.hrfEnabled
  );
  const width = useAppSelector((state: RootState) => state.generation.width);
  const height = useAppSelector((state: RootState) => state.generation.height);

  const handleHrfEnabled = useCallback(
    (e: ChangeEvent<HTMLInputElement>) =>
      dispatch(setHrfEnabled(e.target.checked)),
    [dispatch]
  );

  const label = `Enable High Resolution Fix`;

  return (
    <IAISwitch
      label={label}
      isChecked={hrfEnabled}
      onChange={handleHrfEnabled}
      tooltip={tooltip}
    />
  );
}
