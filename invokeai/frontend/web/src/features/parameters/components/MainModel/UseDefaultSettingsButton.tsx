import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setDefaultSettings } from 'features/parameters/store/actions';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { RiSparklingFill } from 'react-icons/ri';

export const UseDefaultSettingsButton = () => {
  const model = useAppSelector((s) => s.generation.model);
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const handleClickDefaultSettings = useCallback(() => {
    dispatch(setDefaultSettings());
  }, [dispatch]);

  return (
    <IconButton
      icon={<RiSparklingFill />}
      tooltip={t('modelManager.useDefaultSettings')}
      aria-label={t('modelManager.useDefaultSettings')}
      isDisabled={!model}
      onClick={handleClickDefaultSettings}
      size="sm"
      variant="ghost"
    />
  );
};
