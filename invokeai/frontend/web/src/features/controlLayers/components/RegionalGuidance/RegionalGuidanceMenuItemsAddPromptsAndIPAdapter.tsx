import { MenuItem } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import {
  rgIPAdapterAdded,
  rgNegativePromptChanged,
  rgPositivePromptChanged,
  selectCanvasV2Slice,
} from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const RegionalGuidanceMenuItemsAddPromptsAndIPAdapter = memo(() => {
  const { id } = useEntityIdentifierContext();
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const selectValidActions = useMemo(
    () =>
      createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => {
        const rg = canvasV2.regions.entities.find((rg) => rg.id === id);
        return {
          canAddPositivePrompt: rg?.positivePrompt === null,
          canAddNegativePrompt: rg?.negativePrompt === null,
        };
      }),
    [id]
  );
  const validActions = useAppSelector(selectValidActions);
  const addPositivePrompt = useCallback(() => {
    dispatch(rgPositivePromptChanged({ id: id, prompt: '' }));
  }, [dispatch, id]);
  const addNegativePrompt = useCallback(() => {
    dispatch(rgNegativePromptChanged({ id: id, prompt: '' }));
  }, [dispatch, id]);
  const addIPAdapter = useCallback(() => {
    dispatch(rgIPAdapterAdded({ id }));
  }, [dispatch, id]);

  return (
    <>
      <MenuItem onClick={addPositivePrompt} isDisabled={!validActions.canAddPositivePrompt}>
        {t('controlLayers.addPositivePrompt')}
      </MenuItem>
      <MenuItem onClick={addNegativePrompt} isDisabled={!validActions.canAddNegativePrompt}>
        {t('controlLayers.addNegativePrompt')}
      </MenuItem>
      <MenuItem onClick={addIPAdapter}>{t('controlLayers.addIPAdapter')}</MenuItem>
    </>
  );
});

RegionalGuidanceMenuItemsAddPromptsAndIPAdapter.displayName = 'RegionalGuidanceMenuItemsExtra';
