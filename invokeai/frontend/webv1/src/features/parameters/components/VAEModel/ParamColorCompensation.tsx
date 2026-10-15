import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectColorCompensation, setColorCompensation } from 'features/controlLayers/store/paramsSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamColorCompensation = () => {
  const { t } = useTranslation();
  const colorCompensation = useAppSelector(selectColorCompensation);

  const dispatch = useAppDispatch();

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setColorCompensation(e.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="colorCompensation">
        <FormLabel>{t('parameters.colorCompensation')}</FormLabel>
      </InformationalPopover>
      <Switch isChecked={colorCompensation} onChange={handleChange} />
    </FormControl>
  );
};

export default memo(ParamColorCompensation);
