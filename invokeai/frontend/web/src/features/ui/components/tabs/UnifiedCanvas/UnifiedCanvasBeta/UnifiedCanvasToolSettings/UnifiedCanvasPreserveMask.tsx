import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISimpleCheckbox from 'common/components/IAISimpleCheckbox';
import { setShouldPreserveMaskedArea } from 'features/canvas/store/canvasSlice';
import { useTranslation } from 'react-i18next';

export default function UnifiedCanvasPreserveMask() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const shouldPreserveMaskedArea = useAppSelector(
    (state: RootState) => state.canvas.shouldPreserveMaskedArea
  );

  return (
    <IAISimpleCheckbox
      label={t('unifiedCanvas.betaPreserveMasked')}
      isChecked={shouldPreserveMaskedArea}
      onChange={(e) => dispatch(setShouldPreserveMaskedArea(e.target.checked))}
    />
  );
}
