import { useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { useImageSizeContext } from 'features/parameters/components/ImageSize/ImageSizeContext';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamBoundingBoxHeight = () => {
  const { t } = useTranslation();
  const ctx = useImageSizeContext();
  const isStaging = useAppSelector(isStagingSelector);
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const sliderMin = useAppSelector(
    (s) => s.config.sd.boundingBoxHeight.sliderMin
  );
  const sliderMax = useAppSelector(
    (s) => s.config.sd.boundingBoxHeight.sliderMax
  );
  const numberInputMin = useAppSelector(
    (s) => s.config.sd.boundingBoxHeight.numberInputMin
  );
  const numberInputMax = useAppSelector(
    (s) => s.config.sd.boundingBoxHeight.numberInputMax
  );
  const coarseStep = useAppSelector(
    (s) => s.config.sd.boundingBoxHeight.coarseStep
  );
  const fineStep = useAppSelector(
    (s) => s.config.sd.boundingBoxHeight.fineStep
  );
  const onChange = useCallback(
    (v: number) => {
      ctx.heightChanged(v);
    },
    [ctx]
  );

  return (
    <InvControl label={t('parameters.height')} isDisabled={isStaging}>
      <InvSlider
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        value={ctx.height}
        defaultValue={optimalDimension}
        onChange={onChange}
        marks
        withNumberInput
        numberInputMin={numberInputMin}
        numberInputMax={numberInputMax}
      />
    </InvControl>
  );
};

export default memo(ParamBoundingBoxHeight);
