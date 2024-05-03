import { MenuItem } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAddIPAdapterToIPALayer } from 'features/controlLayers/hooks/addLayerHooks';
import {
  isRegionalGuidanceLayer,
  rgLayerNegativePromptChanged,
  rgLayerPositivePromptChanged,
  selectControlLayersSlice,
} from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';
import { assert } from 'tsafe';

type Props = { layerId: string };

export const LayerMenuRGActions = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const [addIPAdapter, isAddIPAdapterDisabled] = useAddIPAdapterToIPALayer(layerId);
  const selectValidActions = useMemo(
    () =>
      createMemoizedSelector(selectControlLayersSlice, (controlLayers) => {
        const layer = controlLayers.present.layers.find((l) => l.id === layerId);
        assert(isRegionalGuidanceLayer(layer), `Layer ${layerId} not found or not an RP layer`);
        return {
          canAddPositivePrompt: layer.positivePrompt === null,
          canAddNegativePrompt: layer.negativePrompt === null,
        };
      }),
    [layerId]
  );
  const validActions = useAppSelector(selectValidActions);
  const addPositivePrompt = useCallback(() => {
    dispatch(rgLayerPositivePromptChanged({ layerId, prompt: '' }));
  }, [dispatch, layerId]);
  const addNegativePrompt = useCallback(() => {
    dispatch(rgLayerNegativePromptChanged({ layerId, prompt: '' }));
  }, [dispatch, layerId]);
  return (
    <>
      <MenuItem onClick={addPositivePrompt} isDisabled={!validActions.canAddPositivePrompt} icon={<PiPlusBold />}>
        {t('controlLayers.addPositivePrompt')}
      </MenuItem>
      <MenuItem onClick={addNegativePrompt} isDisabled={!validActions.canAddNegativePrompt} icon={<PiPlusBold />}>
        {t('controlLayers.addNegativePrompt')}
      </MenuItem>
      <MenuItem onClick={addIPAdapter} icon={<PiPlusBold />} isDisabled={isAddIPAdapterDisabled}>
        {t('controlLayers.addIPAdapter')}
      </MenuItem>
    </>
  );
});

LayerMenuRGActions.displayName = 'LayerMenuRGActions';
