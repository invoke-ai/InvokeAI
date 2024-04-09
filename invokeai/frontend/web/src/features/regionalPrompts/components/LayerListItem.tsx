import { Flex } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { LayerColorPicker } from 'features/regionalPrompts/components/LayerColorPicker';
import { LayerMenu } from 'features/regionalPrompts/components/LayerMenu';
import { LayerVisibilityToggle } from 'features/regionalPrompts/components/LayerVisibilityToggle';
import { RegionalPromptsPrompt } from 'features/regionalPrompts/components/RegionalPromptsPrompt';
import { layerSelected } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { useCallback, useMemo } from 'react';

type Props = {
  id: string;
};

export const LayerListItem = ({ id }: Props) => {
  const dispatch = useAppDispatch();
  const selectedLayer = useAppSelector((s) => s.regionalPrompts.selectedLayer);
  const bg = useMemo(() => (selectedLayer === id ? 'invokeBlue.500' : 'transparent'), [selectedLayer, id]);
  const onClickCapture = useCallback(() => {
    // Must be capture so that the layer is selected before deleting/resetting/etc
    dispatch(layerSelected(id));
  }, [dispatch, id]);
  return (
    <Flex gap={2} onClickCapture={onClickCapture}>
      <Flex w={2} borderRadius="base" bg={bg} flexShrink={0} py={4} />
      <Flex flexDir="column" gap={2}>
        <Flex gap={2}>
          <LayerColorPicker id={id} />
          <LayerVisibilityToggle id={id} />
          <LayerMenu id={id} />
        </Flex>
        <RegionalPromptsPrompt layerId={id} />
      </Flex>
    </Flex>
  );
};
