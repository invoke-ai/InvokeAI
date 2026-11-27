import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  selectZImageSeedVarianceEnabled,
  setZImageSeedVarianceEnabled,
} from 'features/controlLayers/store/paramsSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamZImageSeedVarianceEnabled = () => {
  const { t } = useTranslation();
  const enabled = useAppSelector(selectZImageSeedVarianceEnabled);
  const dispatch = useAppDispatch();

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setZImageSeedVarianceEnabled(e.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="seedVarianceEnhancer">
        <FormLabel>{t('parameters.seedVarianceEnabled')}</FormLabel>
      </InformationalPopover>
      <Switch isChecked={enabled} onChange={handleChange} />
    </FormControl>
  );
};

export default memo(ParamZImageSeedVarianceEnabled);
