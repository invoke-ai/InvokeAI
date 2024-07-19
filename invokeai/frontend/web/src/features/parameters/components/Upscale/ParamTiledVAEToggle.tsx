import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { tiledVAEChanged } from 'features/parameters/store/upscaleSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamTiledVAEToggle = () => {
  const { t } = useTranslation();
  const tiledVAE = useAppSelector((s) => s.upscale.tiledVAE);
  const dispatch = useAppDispatch();

  const handleChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      dispatch(tiledVAEChanged(event.target.checked));
    },
    [dispatch]
  );
  return (
    <FormControl>
      <Switch isChecked={tiledVAE} onChange={handleChange} />
      <FormLabel>{t('upscaling.tiledVAE')}</FormLabel>
    </FormControl>
  );
};
