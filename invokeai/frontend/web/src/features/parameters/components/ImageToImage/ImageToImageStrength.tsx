import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const marks = [0, 0.5, 1];

type Props = {
  value: number;
  onChange: (v: number) => void;
};

const ImageToImageStrength = ({ value, onChange }: Props) => {
  const initial = useAppSelector((s) => s.config.sd.img2imgStrength.initial);
  const sliderMin = useAppSelector((s) => s.config.sd.img2imgStrength.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.img2imgStrength.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.img2imgStrength.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.img2imgStrength.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.img2imgStrength.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.img2imgStrength.fineStep);
  const { t } = useTranslation();

  return (
    <FormControl>
      <InformationalPopover feature="paramDenoisingStrength">
        <FormLabel>{`${t('parameters.denoisingStrength')}`}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        step={coarseStep}
        fineStep={fineStep}
        min={sliderMin}
        max={sliderMax}
        onChange={onChange}
        value={value}
        defaultValue={initial}
        marks={marks}
      />
      <CompositeNumberInput
        step={coarseStep}
        fineStep={fineStep}
        min={numberInputMin}
        max={numberInputMax}
        onChange={onChange}
        value={value}
        defaultValue={initial}
      />
    </FormControl>
  );
};

export default memo(ImageToImageStrength);
