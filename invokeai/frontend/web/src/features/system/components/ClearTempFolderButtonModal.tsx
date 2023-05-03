// import { emptyTempFolder } from 'app/socketio/actions';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIAlertDialog from 'common/components/IAIAlertDialog';
import IAIButton from 'common/components/IAIButton';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import {
  clearCanvasHistory,
  resetCanvas,
} from 'features/canvas/store/canvasSlice';
import { useTranslation } from 'react-i18next';
import { FaTrash } from 'react-icons/fa';

const EmptyTempFolderButtonModal = () => {
  const isStaging = useAppSelector(isStagingSelector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const acceptCallback = () => {
    dispatch(emptyTempFolder());
    dispatch(resetCanvas());
    dispatch(clearCanvasHistory());
  };

  return (
    <IAIAlertDialog
      title={t('unifiedCanvas.emptyTempImageFolder')}
      acceptCallback={acceptCallback}
      acceptButtonText={t('unifiedCanvas.emptyFolder')}
      triggerComponent={
        <IAIButton leftIcon={<FaTrash />} size="sm" isDisabled={isStaging}>
          {t('unifiedCanvas.emptyTempImageFolder')}
        </IAIButton>
      }
    >
      <p>{t('unifiedCanvas.emptyTempImagesFolderMessage')}</p>
      <br />
      <p>{t('unifiedCanvas.emptyTempImagesFolderConfirm')}</p>
    </IAIAlertDialog>
  );
};
export default EmptyTempFolderButtonModal;
