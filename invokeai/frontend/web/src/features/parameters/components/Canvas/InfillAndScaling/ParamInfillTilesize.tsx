import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setInfillTileSize } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamInfillTileSize = () => {
  const dispatch = useAppDispatch();
  const infillTileSize = useAppSelector((s) => s.generation.infillTileSize);
  const initial = useAppSelector((s) => s.config.sd.infillTileSize.initial);
  const sliderMin = useAppSelector((s) => s.config.sd.infillTileSize.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.infillTileSize.sliderMax);
  const numberInputMin = useAppSelector(
    (s) => s.config.sd.infillTileSize.numberInputMin
  );
  const numberInputMax = useAppSelector(
    (s) => s.config.sd.infillTileSize.numberInputMax
  );
  const coarseStep = useAppSelector(
    (s) => s.config.sd.infillTileSize.coarseStep
  );
  const fineStep = useAppSelector((s) => s.config.sd.infillTileSize.fineStep);

  const infillMethod = useAppSelector((s) => s.generation.infillMethod);

  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setInfillTileSize(v));
    },
    [dispatch]
  );

  return (
    <InvControl
      isDisabled={infillMethod !== 'tile'}
      label={t('parameters.tileSize')}
    >
      <InvSlider
        min={sliderMin}
        max={sliderMax}
        numberInputMin={numberInputMin}
        numberInputMax={numberInputMax}
        value={infillTileSize}
        defaultValue={initial}
        onChange={handleChange}
        step={coarseStep}
        fineStep={fineStep}
        withNumberInput
        marks
      />
    </InvControl>
  );
};

export default memo(ParamInfillTileSize);
