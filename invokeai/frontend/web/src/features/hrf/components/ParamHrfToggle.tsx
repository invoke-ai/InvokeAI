import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import type { InvLabelProps } from 'common/components/InvControl/types';
import { InvSwitch } from 'common/components/InvSwitch/wrapper';
import { setHrfEnabled } from 'features/hrf/store/hrfSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamHrfToggle = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const hrfEnabled = useAppSelector((state: RootState) => state.hrf.hrfEnabled);

  const handleHrfEnabled = useCallback(
    (e: ChangeEvent<HTMLInputElement>) =>
      dispatch(setHrfEnabled(e.target.checked)),
    [dispatch]
  );

  return (
    <InvControl label={t('hrf.enableHrf')} labelProps={labelProps} w="full">
      <InvSwitch isChecked={hrfEnabled} onChange={handleHrfEnabled} />
    </InvControl>
  );
};

const labelProps: InvLabelProps = { flexGrow: 1 };

export default memo(ParamHrfToggle);
