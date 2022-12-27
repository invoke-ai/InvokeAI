import { emptyTempFolder } from 'app/socketio/actions';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
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
      title={t('unifiedcanvas:emptyTempImageFolder')}
      acceptCallback={acceptCallback}
      acceptButtonText={t('unifiedcanvas:emptyFolder')}
      triggerComponent={
        <IAIButton leftIcon={<FaTrash />} size={'sm'} isDisabled={isStaging}>
          {t('unifiedcanvas:emptyTempImageFolder')}
        </IAIButton>
      }
    >
      <p>{t('unifiedcanvas:emptyTempImagesFolderMessage')}</p>
      <br />
      <p>{t('unifiedcanvas:emptyTempImagesFolderConfirm')}</p>
    </IAIAlertDialog>
  );
};
export default EmptyTempFolderButtonModal;
