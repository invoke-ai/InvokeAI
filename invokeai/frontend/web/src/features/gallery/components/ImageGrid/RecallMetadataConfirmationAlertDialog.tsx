import { Checkbox, ConfirmationAlertDialog, Flex, FormControl, FormLabel, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { buildUseBoolean } from 'common/hooks/useBoolean';
import {
  selectActiveControlLayerEntities,
  selectActiveInpaintMaskEntities,
  selectActiveRasterLayerEntities,
  selectActiveReferenceImageEntities,
  selectActiveRegionalGuidanceEntities,
} from 'features/controlLayers/store/selectors';
import {
  selectSystemShouldConfirmOnNewSession,
  shouldConfirmOnNewSessionToggled,
} from 'features/system/store/systemSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type RecallMetadataCallback = () => void;

const [useRecallMetadataConfirmationDialog] = buildUseBoolean(false);

let pendingRecallCallback: RecallMetadataCallback | null = null;

export const useRecallMetadataWithConfirmation = () => {
  const dialog = useRecallMetadataConfirmationDialog();
  const shouldConfirm = useAppSelector(selectSystemShouldConfirmOnNewSession);
  
  // Check if there are any active canvas layers that would be affected
  const activeRasterLayers = useAppSelector(selectActiveRasterLayerEntities);
  const activeControlLayers = useAppSelector(selectActiveControlLayerEntities);
  const activeInpaintMasks = useAppSelector(selectActiveInpaintMaskEntities);
  const activeRegionalGuidance = useAppSelector(selectActiveRegionalGuidanceEntities);
  const activeReferenceImages = useAppSelector(selectActiveReferenceImageEntities);
  
  const hasActiveCanvasData = useMemo(() => {
    return (
      activeRasterLayers.length > 0 ||
      activeControlLayers.length > 0 ||
      activeInpaintMasks.length > 0 ||
      activeRegionalGuidance.length > 0 ||
      activeReferenceImages.length > 0
    );
  }, [
    activeRasterLayers.length,
    activeControlLayers.length,
    activeInpaintMasks.length,
    activeRegionalGuidance.length,
    activeReferenceImages.length,
  ]);

  const recallWithConfirmation = useCallback(
    (recallCallback: RecallMetadataCallback) => {
      // If there's no active canvas data or user has disabled confirmations, recall immediately
      if (!hasActiveCanvasData || !shouldConfirm) {
        recallCallback();
        return;
      }

      // Store the callback and show the confirmation dialog
      pendingRecallCallback = recallCallback;
      dialog.setTrue();
    },
    [dialog, hasActiveCanvasData, shouldConfirm]
  );

  return {
    recallWithConfirmation,
    hasActiveCanvasData,
  };
};

export const RecallMetadataConfirmationAlertDialog = memo(() => {
  useAssertSingleton('RecallMetadataConfirmationAlertDialog');
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const dialog = useRecallMetadataConfirmationDialog();
  const shouldConfirm = useAppSelector(selectSystemShouldConfirmOnNewSession);

  const activeRasterLayers = useAppSelector(selectActiveRasterLayerEntities);
  const activeControlLayers = useAppSelector(selectActiveControlLayerEntities);
  const activeInpaintMasks = useAppSelector(selectActiveInpaintMaskEntities);
  const activeRegionalGuidance = useAppSelector(selectActiveRegionalGuidanceEntities);
  const activeReferenceImages = useAppSelector(selectActiveReferenceImageEntities);

  const onConfirm = useCallback(() => {
    if (pendingRecallCallback) {
      pendingRecallCallback();
      pendingRecallCallback = null;
    }
    dialog.setFalse();
  }, [dialog]);

  const onCancel = useCallback(() => {
    pendingRecallCallback = null;
    dialog.setFalse();
  }, [dialog]);

  const onToggleConfirm = useCallback(() => {
    dispatch(shouldConfirmOnNewSessionToggled());
  }, [dispatch]);
  const getCanvasDataSummary = useCallback(() => {
    const items = [];
    if (activeRasterLayers.length > 0) {
      items.push(t('controlLayers.rasterLayer_withCount_other', { count: activeRasterLayers.length }));
    }
    if (activeControlLayers.length > 0) {
      items.push(t('controlLayers.controlLayer_withCount_other', { count: activeControlLayers.length }));
    }
    if (activeInpaintMasks.length > 0) {
      items.push(t('controlLayers.inpaintMask_withCount_other', { count: activeInpaintMasks.length }));
    }
    if (activeRegionalGuidance.length > 0) {
      items.push(t('controlLayers.regionalGuidance_withCount_other', { count: activeRegionalGuidance.length }));
    }
    if (activeReferenceImages.length > 0) {
      items.push(t('controlLayers.globalReferenceImage_withCount_other', { count: activeReferenceImages.length }));
    }
    return items.join(', ');
  }, [activeRasterLayers.length, activeControlLayers.length, activeInpaintMasks.length, activeRegionalGuidance.length, activeReferenceImages.length, t]);

  return (
    <ConfirmationAlertDialog
      isOpen={dialog.isTrue}
      onClose={onCancel}
      title={t('metadata.recallParameters')}
      acceptCallback={onConfirm}
      cancelCallback={onCancel}
      acceptButtonText={t('common.recall')}
      cancelButtonText={t('common.cancel')}
      useInert={false}
    >
      <Flex direction="column" gap={3}>
        <Text>{t('gallery.recallParametersCanvasWarning')}</Text>
        <Text fontWeight="semibold">
          {t('gallery.activeCanvasData', { data: getCanvasDataSummary() })}
        </Text>
        <Text>{t('common.areYouSure')}</Text>
        <FormControl>
          <FormLabel>{t('common.dontAskMeAgain')}</FormLabel>
          <Checkbox isChecked={!shouldConfirm} onChange={onToggleConfirm} />
        </FormControl>
      </Flex>
    </ConfirmationAlertDialog>
  );
});

RecallMetadataConfirmationAlertDialog.displayName = 'RecallMetadataConfirmationAlertDialog';
