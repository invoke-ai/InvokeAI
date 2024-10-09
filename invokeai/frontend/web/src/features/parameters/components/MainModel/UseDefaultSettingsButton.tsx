import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectModel } from 'features/controlLayers/store/paramsSlice';
import { setDefaultSettings } from 'features/parameters/store/actions';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSparkleFill } from 'react-icons/pi';

export const UseDefaultSettingsButton = () => {
  const model = useAppSelector(selectModel);
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const handleClickDefaultSettings = useCallback(() => {
    dispatch(setDefaultSettings());
  }, [dispatch]);

  return (
    <IconButton
      icon={<PiSparkleFill />}
      tooltip={t('modelManager.useDefaultSettings')}
      aria-label={t('modelManager.useDefaultSettings')}
      isDisabled={!model}
      onClick={handleClickDefaultSettings}
      size="sm"
      variant="ghost"
    />
  );
};
