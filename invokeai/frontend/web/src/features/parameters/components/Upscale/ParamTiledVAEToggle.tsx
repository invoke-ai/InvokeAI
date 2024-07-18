import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from '../../../../app/store/storeHooks';
import { useCallback } from 'react';
import { tiledVAEChanged } from '../../store/upscaleSlice';

export const ParamTiledVAEToggle = () => {
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
      <FormLabel>Tiled VAE</FormLabel>
    </FormControl>
  );
};
