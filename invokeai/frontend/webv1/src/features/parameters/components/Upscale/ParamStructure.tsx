import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectStructure, structureChanged } from 'features/parameters/store/upscaleSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const initial = 0;
const sliderMin = -10;
const sliderMax = 10;
const numberInputMin = -10;
const numberInputMax = 10;
const coarseStep = 1;
const fineStep = 1;
const marks = [sliderMin, 0, sliderMax];

const ParamStructure = () => {
  const structure = useAppSelector(selectStructure);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const onChange = useCallback(
    (v: number) => {
      dispatch(structureChanged(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="structure">
        <FormLabel>{t('upscaling.structure')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={structure}
        defaultValue={initial}
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        onChange={onChange}
        marks={marks}
      />
      <CompositeNumberInput
        value={structure}
        defaultValue={initial}
        min={numberInputMin}
        max={numberInputMax}
        step={coarseStep}
        fineStep={fineStep}
        onChange={onChange}
      />
    </FormControl>
  );
};

export default memo(ParamStructure);
