import { MenuItem } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import {
  rgIPAdapterAdded,
  rgNegativePromptChanged,
  rgPositivePromptChanged,
} from 'features/controlLayers/store/canvasV2Slice';
import { selectCanvasV2Slice, selectEntity } from 'features/controlLayers/store/selectors';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const RegionalGuidanceMenuItemsAddPromptsAndIPAdapter = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('regional_guidance');
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const selectValidActions = useMemo(
    () =>
      createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => {
        const entity = selectEntity(canvasV2, entityIdentifier);
        return {
          canAddPositivePrompt: entity?.positivePrompt === null,
          canAddNegativePrompt: entity?.negativePrompt === null,
        };
      }),
    [entityIdentifier]
  );
  const validActions = useAppSelector(selectValidActions);
  const addPositivePrompt = useCallback(() => {
    dispatch(rgPositivePromptChanged({ entityIdentifier, prompt: '' }));
  }, [dispatch, entityIdentifier]);
  const addNegativePrompt = useCallback(() => {
    dispatch(rgNegativePromptChanged({ entityIdentifier, prompt: '' }));
  }, [dispatch, entityIdentifier]);
  const addIPAdapter = useCallback(() => {
    dispatch(rgIPAdapterAdded({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

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
