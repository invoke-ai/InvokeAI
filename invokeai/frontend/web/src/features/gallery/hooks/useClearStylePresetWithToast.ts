import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import {
  activeStylePresetIdChanged,
  selectStylePresetActivePresetId,
} from 'features/stylePresets/store/stylePresetSlice';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const useClearStylePresetWithToast = () => {
  const store = useAppStore();
  const { t } = useTranslation();
  const activeStylePresetId = useAppSelector(selectStylePresetActivePresetId);

  const clearStylePreset = useCallback(() => {
    if (activeStylePresetId) {
      store.dispatch(activeStylePresetIdChanged(null));
      toast({
        status: 'info',
        title: t('stylePresets.promptTemplateCleared'),
      });
    }
  }, [activeStylePresetId, store, t]);

  return clearStylePreset;
};
