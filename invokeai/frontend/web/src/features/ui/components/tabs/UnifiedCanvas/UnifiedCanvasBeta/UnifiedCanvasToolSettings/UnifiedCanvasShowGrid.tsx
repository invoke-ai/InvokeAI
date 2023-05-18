import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAICheckbox from 'common/components/IAICheckbox';
import { setShouldShowGrid } from 'features/canvas/store/canvasSlice';
import { useTranslation } from 'react-i18next';

export default function UnifiedCanvasShowGrid() {
  const shouldShowGrid = useAppSelector(
    (state: RootState) => state.canvas.shouldShowGrid
  );

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  return (
    <IAICheckbox
      label={t('unifiedCanvas.showGrid')}
      isChecked={shouldShowGrid}
      onChange={(e) => dispatch(setShouldShowGrid(e.target.checked))}
    />
  );
}
