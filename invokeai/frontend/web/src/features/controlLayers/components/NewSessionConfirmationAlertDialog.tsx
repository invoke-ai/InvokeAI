import { Checkbox, ConfirmationAlertDialog, Flex, FormControl, FormLabel, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { buildUseBoolean } from 'common/hooks/useBoolean';
import { canvasSessionReset, generateSessionReset } from 'features/controlLayers/store/canvasStagingAreaSlice';
import {
  selectSystemShouldConfirmOnNewSession,
  shouldConfirmOnNewSessionToggled,
} from 'features/system/store/systemSlice';
import { activeTabCanvasRightPanelChanged } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const [useNewGallerySessionDialog] = buildUseBoolean(false);
const [useNewCanvasSessionDialog] = buildUseBoolean(false);

export const useNewGallerySession = () => {
  const dispatch = useAppDispatch();
  const shouldConfirmOnNewSession = useAppSelector(selectSystemShouldConfirmOnNewSession);
  const newSessionDialog = useNewGallerySessionDialog();

  const newGallerySessionImmediate = useCallback(() => {
    dispatch(generateSessionReset());
    dispatch(activeTabCanvasRightPanelChanged('gallery'));
  }, [dispatch]);

  const newGallerySessionWithDialog = useCallback(() => {
    if (shouldConfirmOnNewSession) {
      newSessionDialog.setTrue();
      return;
    }
    newGallerySessionImmediate();
  }, [newGallerySessionImmediate, newSessionDialog, shouldConfirmOnNewSession]);

  return { newGallerySessionImmediate, newGallerySessionWithDialog };
};

export const useNewCanvasSession = () => {
  const dispatch = useAppDispatch();
  const shouldConfirmOnNewSession = useAppSelector(selectSystemShouldConfirmOnNewSession);
  const newSessionDialog = useNewCanvasSessionDialog();

  const newCanvasSessionImmediate = useCallback(() => {
    dispatch(canvasSessionReset());
    dispatch(activeTabCanvasRightPanelChanged('layers'));
  }, [dispatch]);

  const newCanvasSessionWithDialog = useCallback(() => {
    if (shouldConfirmOnNewSession) {
      newSessionDialog.setTrue();
      return;
    }

    newCanvasSessionImmediate();
  }, [newCanvasSessionImmediate, newSessionDialog, shouldConfirmOnNewSession]);

  return { newCanvasSessionImmediate, newCanvasSessionWithDialog };
};

export const NewGallerySessionDialog = memo(() => {
  useAssertSingleton('NewGallerySessionDialog');
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const dialog = useNewGallerySessionDialog();
  const { newGallerySessionImmediate } = useNewGallerySession();

  const shouldConfirmOnNewSession = useAppSelector(selectSystemShouldConfirmOnNewSession);
  const onToggleConfirm = useCallback(() => {
    dispatch(shouldConfirmOnNewSessionToggled());
  }, [dispatch]);

  return (
    <ConfirmationAlertDialog
      isOpen={dialog.isTrue}
      onClose={dialog.setFalse}
      title={t('controlLayers.newGallerySession')}
      acceptCallback={newGallerySessionImmediate}
      acceptButtonText={t('common.ok')}
      useInert={false}
    >
      <Flex direction="column" gap={3}>
        <Text>{t('controlLayers.newGallerySessionDesc')}</Text>
        <Text>{t('common.areYouSure')}</Text>
        <FormControl>
          <FormLabel>{t('common.dontAskMeAgain')}</FormLabel>
          <Checkbox isChecked={!shouldConfirmOnNewSession} onChange={onToggleConfirm} />
        </FormControl>
      </Flex>
    </ConfirmationAlertDialog>
  );
});

NewGallerySessionDialog.displayName = 'NewGallerySessionDialog';

export const NewCanvasSessionDialog = memo(() => {
  useAssertSingleton('NewCanvasSessionDialog');
  const { t } = useTranslation();

  const dispatch = useAppDispatch();

  const dialog = useNewCanvasSessionDialog();
  const { newCanvasSessionImmediate } = useNewCanvasSession();

  const shouldConfirmOnNewSession = useAppSelector(selectSystemShouldConfirmOnNewSession);
  const onToggleConfirm = useCallback(() => {
    dispatch(shouldConfirmOnNewSessionToggled());
  }, [dispatch]);

  return (
    <ConfirmationAlertDialog
      isOpen={dialog.isTrue}
      onClose={dialog.setFalse}
      title={t('controlLayers.newCanvasSession')}
      acceptCallback={newCanvasSessionImmediate}
      acceptButtonText={t('common.ok')}
      useInert={false}
    >
      <Flex direction="column" gap={3}>
        <Text>{t('controlLayers.newCanvasSessionDesc')}</Text>
        <Text>{t('common.areYouSure')}</Text>
        <FormControl>
          <FormLabel>{t('common.dontAskMeAgain')}</FormLabel>
          <Checkbox isChecked={!shouldConfirmOnNewSession} onChange={onToggleConfirm} />
        </FormControl>
      </Flex>
    </ConfirmationAlertDialog>
  );
});

NewCanvasSessionDialog.displayName = 'NewCanvasSessionDialog';
