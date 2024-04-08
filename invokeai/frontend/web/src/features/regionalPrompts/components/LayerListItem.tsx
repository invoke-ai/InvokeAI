import { Flex } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { DeleteLayerButton } from 'features/regionalPrompts/components/DeleteLayerButton';
import { LayerColorPicker } from 'features/regionalPrompts/components/LayerColorPicker';
import { RegionalPromptsPrompt } from 'features/regionalPrompts/components/RegionalPromptsPrompt';
import { ResetLayerButton } from 'features/regionalPrompts/components/ResetLayerButton';
import { layerSelected } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { useCallback, useMemo } from 'react';

type Props = {
  id: string;
};

export const LayerListItem = ({ id }: Props) => {
  const dispatch = useAppDispatch();
  const selectedLayer = useAppSelector((s) => s.regionalPrompts.selectedLayer);
  const border = useMemo(() => (selectedLayer === id ? '1px solid red' : 'none'), [selectedLayer, id]);
  const onClickCapture = useCallback(() => {
    // Must be capture so that the layer is selected before deleting/resetting/etc
    dispatch(layerSelected(id));
  }, [dispatch, id]);
  return (
    <Flex flexDir="column" onClickCapture={onClickCapture} border={border}>
      <Flex gap={2}>
        <ResetLayerButton id={id} />
        <DeleteLayerButton id={id} />
        <LayerColorPicker id={id} />
      </Flex>
      <RegionalPromptsPrompt layerId={id} />
    </Flex>
  );
};
