import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setHrfEnabled } from 'features/hrf/store/hrfSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamHrfToggle = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const hrfEnabled = useAppSelector((s) => s.hrf.hrfEnabled);

  const handleHrfEnabled = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(setHrfEnabled(e.target.checked)),
    [dispatch]
  );

  return (
    <FormControl w="full">
      <FormLabel flexGrow={1}>{t('hrf.enableHrf')}</FormLabel>
      <Switch isChecked={hrfEnabled} onChange={handleHrfEnabled} />
    </FormControl>
  );
};

export default memo(ParamHrfToggle);
