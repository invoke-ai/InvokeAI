import { Button, Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAddIPAdapterToRGLayer } from 'features/controlLayers/hooks/addLayerHooks';
import {
  rgNegativePromptChanged,
  rgPositivePromptChanged,
  selectCanvasV2Slice,
} from 'features/controlLayers/store/canvasV2Slice';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

type AddPromptButtonProps = {
  id: string;
};

export const AddPromptButtons = ({ id }: AddPromptButtonProps) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const [addIPAdapter, isAddIPAdapterDisabled] = useAddIPAdapterToRGLayer(id);
  const selectValidActions = useMemo(
    () =>
      createMemoizedSelector(selectCanvasV2Slice, (caState) => {
        const rg = caState.regions.find((rg) => rg.id === id);
        return {
          canAddPositivePrompt: rg?.positivePrompt === null,
          canAddNegativePrompt: rg?.negativePrompt === null,
        };
      }),
    [id]
  );
  const validActions = useAppSelector(selectValidActions);
  const addPositivePrompt = useCallback(() => {
    dispatch(rgPositivePromptChanged({ id, prompt: '' }));
  }, [dispatch, id]);
  const addNegativePrompt = useCallback(() => {
    dispatch(rgNegativePromptChanged({ id, prompt: '' }));
  }, [dispatch, id]);

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
