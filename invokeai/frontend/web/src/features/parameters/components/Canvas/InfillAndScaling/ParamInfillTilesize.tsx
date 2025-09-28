import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectInfillMethod,
  selectInfillTileSize,
  setInfillTileSize,
  useParamsDispatch,
} from 'features/controlLayers/store/paramsSlice';
import { selectInfillTileSizeConfig } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamInfillTileSize = () => {
  const dispatchParams = useParamsDispatch();
  const infillTileSize = useAppSelector(selectInfillTileSize);
  const config = useAppSelector(selectInfillTileSizeConfig);
  const infillMethod = useAppSelector(selectInfillMethod);

  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatchParams(setInfillTileSize, v);
    },
    [dispatchParams]
  );

  return (
    <FormControl isDisabled={infillMethod !== 'tile'}>
      <FormLabel>{t('parameters.tileSize')}</FormLabel>
      <CompositeSlider
        value={infillTileSize}
        onChange={handleChange}
        defaultValue={config.initial}
        min={config.sliderMin}
        max={config.sliderMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
        marks
      />
      <CompositeNumberInput
        value={infillTileSize}
        onChange={handleChange}
        defaultValue={config.initial}
        min={config.numberInputMin}
        max={config.numberInputMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
      />
    </FormControl>
  );
};

export default memo(ParamInfillTileSize);
