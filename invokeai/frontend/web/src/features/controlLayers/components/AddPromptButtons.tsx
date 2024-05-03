import { Button, Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAddIPAdapterToIPALayer } from 'features/controlLayers/hooks/addLayerHooks';
import {
  isRegionalGuidanceLayer,
  rgLayerNegativePromptChanged,
  rgLayerPositivePromptChanged,
  selectControlLayersSlice,
} from 'features/controlLayers/store/controlLayersSlice';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';
import { assert } from 'tsafe';
type AddPromptButtonProps = {
  layerId: string;
};

export const AddPromptButtons = ({ layerId }: AddPromptButtonProps) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
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
    <Flex w="full" p={2} justifyContent="space-between">
      <Button
        size="sm"
        variant="ghost"
        leftIcon={<PiPlusBold />}
        onClick={addPositivePrompt}
        isDisabled={!validActions.canAddPositivePrompt}
      >
        {t('common.positivePrompt')}
      </Button>
      <Button
        size="sm"
        variant="ghost"
        leftIcon={<PiPlusBold />}
        onClick={addNegativePrompt}
        isDisabled={!validActions.canAddNegativePrompt}
      >
        {t('common.negativePrompt')}
      </Button>
      <Button
        size="sm"
        variant="ghost"
        leftIcon={<PiPlusBold />}
        onClick={addIPAdapter}
        isDisabled={isAddIPAdapterDisabled}
      >
        {t('common.ipAdapter')}
      </Button>
    </Flex>
  );
};
