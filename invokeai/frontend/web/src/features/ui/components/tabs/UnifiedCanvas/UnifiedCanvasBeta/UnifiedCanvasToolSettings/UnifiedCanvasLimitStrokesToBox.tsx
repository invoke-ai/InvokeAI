import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAICheckbox from 'common/components/IAICheckbox';
import { setShouldRestrictStrokesToBox } from 'features/canvas/store/canvasSlice';
import { useTranslation } from 'react-i18next';

export default function UnifiedCanvasLimitStrokesToBox() {
  const dispatch = useAppDispatch();

  const shouldRestrictStrokesToBox = useAppSelector(
    (state: RootState) => state.canvas.shouldRestrictStrokesToBox
  );

  const { t } = useTranslation();

  return (
    <IAICheckbox
      label={t('unifiedCanvas.betaLimitToBox')}
      isChecked={shouldRestrictStrokesToBox}
      onChange={(e) =>
        dispatch(setShouldRestrictStrokesToBox(e.target.checked))
      }
    />
  );
}
