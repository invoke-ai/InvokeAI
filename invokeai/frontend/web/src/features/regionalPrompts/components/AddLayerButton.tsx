import { Button } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { layerAdded } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const AddLayerButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    dispatch(layerAdded('regionalPromptLayer'));
  }, [dispatch]);

  return <Button onClick={onClick}>{t('regionalPrompts.addLayer')}</Button>;
});

AddLayerButton.displayName = 'AddLayerButton';
