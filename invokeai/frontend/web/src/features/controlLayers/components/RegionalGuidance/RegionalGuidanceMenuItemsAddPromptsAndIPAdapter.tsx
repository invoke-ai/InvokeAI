import { MenuItem } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import {
  buildSelectValidRegionalGuidanceActions,
  useAddRegionalGuidanceIPAdapter,
  useAddRegionalGuidanceNegativePrompt,
  useAddRegionalGuidancePositivePrompt,
} from 'features/controlLayers/hooks/addLayerHooks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const RegionalGuidanceMenuItemsAddPromptsAndIPAdapter = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('regional_guidance');
  const { t } = useTranslation();
  const isBusy = useCanvasIsBusy();
  const addRegionalGuidanceIPAdapter = useAddRegionalGuidanceIPAdapter(entityIdentifier);
  const addRegionalGuidancePositivePrompt = useAddRegionalGuidancePositivePrompt(entityIdentifier);
  const addRegionalGuidanceNegativePrompt = useAddRegionalGuidanceNegativePrompt(entityIdentifier);
  const selectValidActions = useMemo(
    () => buildSelectValidRegionalGuidanceActions(entityIdentifier),
    [entityIdentifier]
  );
  const validActions = useAppSelector(selectValidActions);

  return (
    <>
      <MenuItem onClick={addRegionalGuidancePositivePrompt} isDisabled={!validActions.canAddPositivePrompt || isBusy}>
        {t('controlLayers.addPositivePrompt')}
      </MenuItem>
      <MenuItem onClick={addRegionalGuidanceNegativePrompt} isDisabled={!validActions.canAddNegativePrompt || isBusy}>
        {t('controlLayers.addNegativePrompt')}
      </MenuItem>
      <MenuItem onClick={addRegionalGuidanceIPAdapter} isDisabled={isBusy}>
        {t('controlLayers.addReferenceImage')}
      </MenuItem>
    </>
  );
});

RegionalGuidanceMenuItemsAddPromptsAndIPAdapter.displayName = 'RegionalGuidanceMenuItemsExtra';
