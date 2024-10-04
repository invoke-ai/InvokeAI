import { Checkbox, ConfirmationAlertDialog, Flex, FormControl, FormLabel, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { buildUseBoolean } from 'common/hooks/useBoolean';
import { newCanvasSessionRequested, newGallerySessionRequested } from 'features/controlLayers/store/actions';
import {
  selectCanvasRightPanelGalleryTab,
  selectCanvasRightPanelLayersTab,
} from 'features/controlLayers/store/ephemeral';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import {
  selectSystemShouldConfirmOnNewSession,
  shouldConfirmOnNewSessionToggled,
} from 'features/system/store/systemSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const [useNewGallerySessionDialog] = buildUseBoolean(false);
const [useNewCanvasSessionDialog] = buildUseBoolean(false);

export const useNewGallerySession = () => {
  const dispatch = useAppDispatch();
  const imageViewer = useImageViewer();
  const shouldConfirmOnNewSession = useAppSelector(selectSystemShouldConfirmOnNewSession);
  const newSessionDialog = useNewGallerySessionDialog();

  const newGallerySessionImmediate = useCallback(() => {
    dispatch(newGallerySessionRequested());
    imageViewer.open();
    selectCanvasRightPanelGalleryTab();
  }, [dispatch, imageViewer]);

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
  const imageViewer = useImageViewer();
  const shouldConfirmOnNewSession = useAppSelector(selectSystemShouldConfirmOnNewSession);
  const newSessionDialog = useNewCanvasSessionDialog();

  const newCanvasSessionImmediate = useCallback(() => {
    dispatch(newCanvasSessionRequested());
    imageViewer.close();
    selectCanvasRightPanelLayersTab();
  }, [dispatch, imageViewer]);

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
