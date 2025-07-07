import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectSaveAllStagingImagesToGallery,
  settingsSaveAllStagingImagesToGalleryToggled,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsSaveAllStagingImagesToGalleryCheckbox = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const saveAllStagingImagesToGallery = useAppSelector(selectSaveAllStagingImagesToGallery);
  const onChange = useCallback(() => {
    dispatch(settingsSaveAllStagingImagesToGalleryToggled());
  }, [dispatch]);
  return (
    <FormControl w="full">
      <FormLabel flexGrow={1}>{t('controlLayers.saveAllStagingImagesToGallery')}</FormLabel>
      <Checkbox isChecked={saveAllStagingImagesToGallery} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsSaveAllStagingImagesToGalleryCheckbox.displayName = 'CanvasSettingsSaveAllStagingImagesToGalleryCheckbox';
