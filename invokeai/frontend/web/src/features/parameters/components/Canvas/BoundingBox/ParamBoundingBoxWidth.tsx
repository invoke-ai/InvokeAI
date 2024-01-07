import { useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { useImageSizeContext } from 'features/parameters/components/ImageSize/ImageSizeContext';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamBoundingBoxWidth = () => {
  const { t } = useTranslation();
  const ctx = useImageSizeContext();
  const isStaging = useAppSelector(isStagingSelector);
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const sliderMin = useAppSelector(
    (s) => s.config.sd.boundingBoxWidth.sliderMin
  );
  const sliderMax = useAppSelector(
    (s) => s.config.sd.boundingBoxWidth.sliderMax
  );
  const numberInputMin = useAppSelector(
    (s) => s.config.sd.boundingBoxWidth.numberInputMin
  );
  const numberInputMax = useAppSelector(
    (s) => s.config.sd.boundingBoxWidth.numberInputMax
  );
  const coarseStep = useAppSelector(
    (s) => s.config.sd.boundingBoxWidth.coarseStep
  );
  const fineStep = useAppSelector((s) => s.config.sd.boundingBoxWidth.fineStep);
  const onChange = useCallback(
    (v: number) => {
      ctx.widthChanged(v);
    },
    [ctx]
  );

  return (
    <InvControl label={t('parameters.width')} isDisabled={isStaging}>
      <InvSlider
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        value={ctx.width}
        defaultValue={optimalDimension}
        onChange={onChange}
        withNumberInput
        numberInputMin={numberInputMin}
        numberInputMax={numberInputMax}
        marks
      />
    </InvControl>
  );
};

export default memo(ParamBoundingBoxWidth);
