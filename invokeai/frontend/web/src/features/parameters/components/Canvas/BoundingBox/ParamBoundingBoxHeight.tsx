import { useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import {
  CANVAS_GRID_SIZE_COARSE,
  CANVAS_GRID_SIZE_FINE,
} from 'features/canvas/store/constants';
import { useImageSizeContext } from 'features/parameters/components/ImageSize/ImageSizeContext';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamBoundingBoxWidth = () => {
  const isStaging = useAppSelector(isStagingSelector);
  const initial = useAppSelector(selectOptimalDimension);
  const ctx = useImageSizeContext();
  const { t } = useTranslation();

  const onChange = useCallback(
    (v: number) => {
      ctx.heightChanged(v);
    },
    [ctx]
  );

  return (
    <InvControl label={t('parameters.height')} isDisabled={isStaging}>
      <InvSlider
        min={64}
        max={1536}
        step={CANVAS_GRID_SIZE_COARSE}
        fineStep={CANVAS_GRID_SIZE_FINE}
        value={ctx.height}
        defaultValue={initial}
        onChange={onChange}
        marks
        withNumberInput
        numberInputMax={4096}
      />
    </InvControl>
  );
};

export default memo(ParamBoundingBoxWidth);
