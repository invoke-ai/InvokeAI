import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISimpleCheckbox from 'common/components/IAISimpleCheckbox';
import { setIsMaskEnabled } from 'features/canvas/store/canvasSlice';
import { useTranslation } from 'react-i18next';

export default function UnifiedCanvasEnableMask() {
  const isMaskEnabled = useAppSelector(
    (state: RootState) => state.canvas.isMaskEnabled
  );

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleToggleEnableMask = () =>
    dispatch(setIsMaskEnabled(!isMaskEnabled));

  return (
    <IAISimpleCheckbox
      label={`${t('unifiedCanvas.enableMask')} (H)`}
      isChecked={isMaskEnabled}
      onChange={handleToggleEnableMask}
    />
  );
}
