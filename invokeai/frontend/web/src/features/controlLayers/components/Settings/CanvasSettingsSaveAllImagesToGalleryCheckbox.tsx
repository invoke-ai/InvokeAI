import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectSaveAllImagesToGallery,
  settingsSaveAllImagesToGalleryToggled,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsSaveAllImagesToGalleryCheckbox = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const saveAllImagesToGallery = useAppSelector(selectSaveAllImagesToGallery);
  const onChange = useCallback(() => {
    dispatch(settingsSaveAllImagesToGalleryToggled());
  }, [dispatch]);
  return (
    <FormControl w="full">
      <FormLabel flexGrow={1}>{t('controlLayers.settings.saveAllImagesToGallery.label')}</FormLabel>
      <Checkbox isChecked={saveAllImagesToGallery} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsSaveAllImagesToGalleryCheckbox.displayName = 'CanvasSettingsSaveAllImagesToGalleryCheckbox';
