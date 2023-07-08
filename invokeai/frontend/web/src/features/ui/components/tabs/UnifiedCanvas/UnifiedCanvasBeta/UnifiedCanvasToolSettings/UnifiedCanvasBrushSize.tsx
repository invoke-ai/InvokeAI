import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { setBrushSize } from 'features/canvas/store/canvasSlice';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';

export default function UnifiedCanvasBrushSize() {
  const dispatch = useAppDispatch();

  const brushSize = useAppSelector(
    (state: RootState) => state.canvas.brushSize
  );

  const { t } = useTranslation();

  const isStaging = useAppSelector(isStagingSelector);

  useHotkeys(
    ['BracketLeft'],
    () => {
      dispatch(setBrushSize(Math.max(brushSize - 5, 5)));
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [brushSize]
  );

  useHotkeys(
    ['BracketRight'],
    () => {
      dispatch(setBrushSize(Math.min(brushSize + 5, 500)));
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [brushSize]
  );

  return (
    <IAISlider
      label={t('unifiedCanvas.brushSize')}
      value={brushSize}
      withInput
      onChange={(newSize) => dispatch(setBrushSize(newSize))}
      sliderNumberInputProps={{ max: 500 }}
      isCompact
    />
  );
}
