import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISimpleCheckbox from 'common/components/IAISimpleCheckbox';
import { setShouldSnapToGrid } from 'features/canvas/store/canvasSlice';
import { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

export default function UnifiedCanvasSnapToGrid() {
  const shouldSnapToGrid = useAppSelector(
    (state: RootState) => state.canvas.shouldSnapToGrid
  );

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChangeShouldSnapToGrid = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldSnapToGrid(e.target.checked));

  return (
    <IAISimpleCheckbox
      label={`${t('unifiedCanvas.snapToGrid')} (N)`}
      isChecked={shouldSnapToGrid}
      onChange={handleChangeShouldSnapToGrid}
    />
  );
}
