import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectKrea2RebalanceEnabled, setKrea2RebalanceEnabled } from 'features/controlLayers/store/paramsSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamKrea2RebalanceEnabled = () => {
  const { t } = useTranslation();
  const enabled = useAppSelector(selectKrea2RebalanceEnabled);
  const dispatch = useAppDispatch();

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setKrea2RebalanceEnabled(e.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="krea2ConditioningRebalance">
        <FormLabel>{t('parameters.krea2RebalanceEnabled')}</FormLabel>
      </InformationalPopover>
      <Switch isChecked={enabled} onChange={handleChange} />
    </FormControl>
  );
};

export default memo(ParamKrea2RebalanceEnabled);
