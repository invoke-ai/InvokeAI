import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setHrfEnabled } from 'features/parameters/store/generationSlice';
import { ChangeEvent, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export default function ParamHrfToggle() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

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
      label={t('hrf.enableHrf')}
      isChecked={hrfEnabled}
      onChange={handleHrfEnabled}
      tooltip={t('hrf.enableHrfTooltip')}
    />
  );
}
