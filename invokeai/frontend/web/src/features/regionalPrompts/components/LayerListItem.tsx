import { Flex, Spacer } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { rgbColorToString } from 'features/canvas/util/colorToString';
import { LayerColorPicker } from 'features/regionalPrompts/components/LayerColorPicker';
import { LayerMenu } from 'features/regionalPrompts/components/LayerMenu';
import { LayerVisibilityToggle } from 'features/regionalPrompts/components/LayerVisibilityToggle';
import { RegionalPromptsPrompt } from 'features/regionalPrompts/components/RegionalPromptsPrompt';
import { layerSelected } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { useCallback } from 'react';

type Props = {
  id: string;
};

export const LayerListItem = ({ id }: Props) => {
  const dispatch = useAppDispatch();
  const color = useAppSelector((s) => {
    const color = s.regionalPrompts.layers.find((l) => l.id === id)?.color;
    if (color) {
      return rgbColorToString(color);
    }
    return 'base.700';
  });
  const onClickCapture = useCallback(() => {
    // Must be capture so that the layer is selected before deleting/resetting/etc
    dispatch(layerSelected(id));
  }, [dispatch, id]);
  return (
    <Flex gap={2} onClickCapture={onClickCapture} bg={color} borderRadius="base" p="1px" ps={2}>
      <Flex flexDir="column" gap={2} w="full" bg="base.850" borderRadius="base" p={2}>
        <Flex gap={2}>
          <LayerColorPicker id={id} />
          <LayerVisibilityToggle id={id} />
          <Spacer />
          <LayerMenu id={id} />
        </Flex>
        <RegionalPromptsPrompt layerId={id} />
      </Flex>
    </Flex>
  );
};
