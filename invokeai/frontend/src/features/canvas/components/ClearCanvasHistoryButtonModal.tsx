import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIAlertDialog from 'common/components/IAIAlertDialog';
import IAIButton from 'common/components/IAIButton';
import { clearCanvasHistory } from 'features/canvas/store/canvasSlice';
import { useTranslation } from 'react-i18next';
import { FaTrash } from 'react-icons/fa';
import { isStagingSelector } from '../store/canvasSelectors';

const ClearCanvasHistoryButtonModal = () => {
  const isStaging = useAppSelector(isStagingSelector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  return (
    <IAIAlertDialog
      title={t('unifiedcanvas:clearCanvasHistory')}
      acceptCallback={() => dispatch(clearCanvasHistory())}
      acceptButtonText={t('unifiedcanvas:clearHistory')}
      triggerComponent={
        <IAIButton size={'sm'} leftIcon={<FaTrash />} isDisabled={isStaging}>
          {t('unifiedcanvas:clearCanvasHistory')}
        </IAIButton>
      }
    >
      <p>{t('unifiedcanvas:clearCanvasHistoryMessage')}</p>
      <br />
      <p>{t('unifiedcanvas:clearCanvasHistoryConfirm')}</p>
    </IAIAlertDialog>
  );
};
export default ClearCanvasHistoryButtonModal;
