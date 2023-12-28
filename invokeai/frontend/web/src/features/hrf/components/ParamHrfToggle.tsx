import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSwitch } from 'common/components/InvSwitch/wrapper';
import { setHrfEnabled } from 'features/hrf/store/hrfSlice';
import type { ChangeEvent } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export default function ParamHrfToggle() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const hrfEnabled = useAppSelector((state: RootState) => state.hrf.hrfEnabled);

  const handleHrfEnabled = useCallback(
    (e: ChangeEvent<HTMLInputElement>) =>
      dispatch(setHrfEnabled(e.target.checked)),
    [dispatch]
  );

  return (
    <InvControl
      label={t('hrf.enableHrf')}
      labelProps={{ flexGrow: 1 }}
      w="full"
    >
      <InvSwitch isChecked={hrfEnabled} onChange={handleHrfEnabled} />
    </InvControl>
  );
}
