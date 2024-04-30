import { MenuItem } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { guidanceLayerIPAdapterAdded } from 'app/store/middleware/listenerMiddleware/listeners/regionalControlToControlAdapterBridge';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  isMaskedGuidanceLayer,
  maskLayerNegativePromptChanged,
  maskLayerPositivePromptChanged,
  selectRegionalPromptsSlice,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';
import { assert } from 'tsafe';

type Props = { layerId: string };

export const RPLayerMenuMaskedGuidanceActions = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const selectValidActions = useMemo(
    () =>
      createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
        const layer = regionalPrompts.present.layers.find((l) => l.id === layerId);
        assert(isMaskedGuidanceLayer(layer), `Layer ${layerId} not found or not an RP layer`);
        return {
          canAddPositivePrompt: layer.positivePrompt === null,
          canAddNegativePrompt: layer.negativePrompt === null,
        };
      }),
    [layerId]
  );
  const validActions = useAppSelector(selectValidActions);
  const addPositivePrompt = useCallback(() => {
    dispatch(maskLayerPositivePromptChanged({ layerId, prompt: '' }));
  }, [dispatch, layerId]);
  const addNegativePrompt = useCallback(() => {
    dispatch(maskLayerNegativePromptChanged({ layerId, prompt: '' }));
  }, [dispatch, layerId]);
  const addIPAdapter = useCallback(() => {
    dispatch(guidanceLayerIPAdapterAdded(layerId));
  }, [dispatch, layerId]);
  return (
    <>
      <MenuItem onClick={addPositivePrompt} isDisabled={!validActions.canAddPositivePrompt} icon={<PiPlusBold />}>
        {t('regionalPrompts.addPositivePrompt')}
      </MenuItem>
      <MenuItem onClick={addNegativePrompt} isDisabled={!validActions.canAddNegativePrompt} icon={<PiPlusBold />}>
        {t('regionalPrompts.addNegativePrompt')}
      </MenuItem>
      <MenuItem onClick={addIPAdapter} icon={<PiPlusBold />}>
        {t('regionalPrompts.addIPAdapter')}
      </MenuItem>
    </>
  );
});

RPLayerMenuMaskedGuidanceActions.displayName = 'RPLayerMenuMaskedGuidanceActions';
