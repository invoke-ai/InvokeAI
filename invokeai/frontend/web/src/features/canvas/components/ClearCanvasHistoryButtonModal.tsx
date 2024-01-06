import { useDisclosure } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvConfirmationAlertDialog } from 'common/components/InvConfirmationAlertDialog/InvConfirmationAlertDialog';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { clearCanvasHistory } from 'features/canvas/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleFill } from 'react-icons/pi'

const ClearCanvasHistoryButtonModal = () => {
  const isStaging = useAppSelector(isStagingSelector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const acceptCallback = useCallback(
    () => dispatch(clearCanvasHistory()),
    [dispatch]
  );

  return (
    <>
      <InvButton
        onClick={onOpen}
        size="sm"
        leftIcon={<PiTrashSimpleFill />}
        isDisabled={isStaging}
      >
        {t('unifiedCanvas.clearCanvasHistory')}
      </InvButton>
      <InvConfirmationAlertDialog
        isOpen={isOpen}
        onClose={onClose}
        title={t('unifiedCanvas.clearCanvasHistory')}
        acceptCallback={acceptCallback}
        acceptButtonText={t('unifiedCanvas.clearHistory')}
      >
        <p>{t('unifiedCanvas.clearCanvasHistoryMessage')}</p>
        <br />
        <p>{t('unifiedCanvas.clearCanvasHistoryConfirm')}</p>
      </InvConfirmationAlertDialog>
    </>
  );
};
export default memo(ClearCanvasHistoryButtonModal);
