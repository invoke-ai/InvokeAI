import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectSeamlessYAxis, setSeamlessYAxis, useParamsDispatch } from 'features/controlLayers/store/paramsSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSeamlessYAxis = () => {
  const { t } = useTranslation();
  const seamlessYAxis = useAppSelector(selectSeamlessYAxis);
  const dispatchParams = useParamsDispatch();
  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatchParams(setSeamlessYAxis, e.target.checked);
    },
    [dispatchParams]
  );

  return (
    <FormControl>
      <InformationalPopover feature="seamlessTilingYAxis">
        <FormLabel>{t('parameters.seamlessYAxis')}</FormLabel>
      </InformationalPopover>
      <Switch isChecked={seamlessYAxis} onChange={handleChange} />
    </FormControl>
  );
};

export default memo(ParamSeamlessYAxis);
