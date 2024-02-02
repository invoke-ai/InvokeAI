import { Button, ConfirmationAlertDialog, useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { clearCanvasHistory } from 'features/canvas/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleFill } from 'react-icons/pi';

const ClearCanvasHistoryButtonModal = () => {
  const isStaging = useAppSelector(isStagingSelector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const acceptCallback = useCallback(() => dispatch(clearCanvasHistory()), [dispatch]);

  return (
    <>
      <Button onClick={onOpen} size="sm" leftIcon={<PiTrashSimpleFill />} isDisabled={isStaging}>
        {t('unifiedCanvas.clearCanvasHistory')}
      </Button>
      <ConfirmationAlertDialog
        isOpen={isOpen}
        onClose={onClose}
        title={t('unifiedCanvas.clearCanvasHistory')}
        acceptCallback={acceptCallback}
        acceptButtonText={t('unifiedCanvas.clearHistory')}
      >
        <p>{t('unifiedCanvas.clearCanvasHistoryMessage')}</p>
        <br />
        <p>{t('unifiedCanvas.clearCanvasHistoryConfirm')}</p>
      </ConfirmationAlertDialog>
    </>
  );
};
export default memo(ClearCanvasHistoryButtonModal);
