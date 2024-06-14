import { MenuItem } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAddIPAdapterToIPALayer } from 'features/controlLayers/hooks/addLayerHooks';
import {
  regionalGuidanceNegativePromptChanged,
  regionalGuidancePositivePromptChanged,
  selectCanvasV2Slice,
} from 'features/controlLayers/store/controlLayersSlice';
import { isRegionalGuidanceLayer } from 'features/controlLayers/store/types';
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
      createMemoizedSelector(selectCanvasV2Slice, (controlLayers) => {
        const layer = canvasV2.layers.find((l) => l.id === layerId);
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
    dispatch(regionalGuidancePositivePromptChanged({ layerId, prompt: '' }));
  }, [dispatch, layerId]);
  const addNegativePrompt = useCallback(() => {
    dispatch(regionalGuidanceNegativePromptChanged({ layerId, prompt: '' }));
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
