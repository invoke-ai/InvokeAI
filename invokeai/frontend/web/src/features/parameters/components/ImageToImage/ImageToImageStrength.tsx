import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectImg2imgStrengthConfig } from 'features/system/store/configSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const marks = [0, 0.5, 1];

type Props = {
  value: number;
  onChange: (v: number) => void;
};

const ImageToImageStrength = ({ value, onChange }: Props) => {
  const config = useAppSelector(selectImg2imgStrengthConfig);
  const { t } = useTranslation();

  return (
    <FormControl>
      <InformationalPopover feature="paramDenoisingStrength">
        <FormLabel>{`${t('parameters.denoisingStrength')}`}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        step={config.coarseStep}
        fineStep={config.fineStep}
        min={config.sliderMin}
        max={config.sliderMax}
        defaultValue={config.initial}
        onChange={onChange}
        value={value}
        marks={marks}
      />
      <CompositeNumberInput
        step={config.coarseStep}
        fineStep={config.fineStep}
        min={config.numberInputMin}
        max={config.numberInputMax}
        defaultValue={config.initial}
        onChange={onChange}
        value={value}
      />
    </FormControl>
  );
};

export default memo(ImageToImageStrength);
