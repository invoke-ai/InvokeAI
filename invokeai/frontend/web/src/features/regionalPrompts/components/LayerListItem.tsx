import { Flex, Spacer } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { rgbaColorToString } from 'features/canvas/util/colorToString';
import LayerAutoNegativeCombobox from 'features/regionalPrompts/components/LayerAutoNegativeCombobox';
import { LayerColorPicker } from 'features/regionalPrompts/components/LayerColorPicker';
import { LayerMenu } from 'features/regionalPrompts/components/LayerMenu';
import { LayerVisibilityToggle } from 'features/regionalPrompts/components/LayerVisibilityToggle';
import { RegionalPromptsNegativePrompt } from 'features/regionalPrompts/components/RegionalPromptsNegativePrompt';
import { RegionalPromptsPositivePrompt } from 'features/regionalPrompts/components/RegionalPromptsPositivePrompt';
import { layerSelected } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback } from 'react';

type Props = {
  id: string;
};

export const LayerListItem = memo(({ id }: Props) => {
  const dispatch = useAppDispatch();
  const selectedLayer = useAppSelector((s) => s.regionalPrompts.present.selectedLayer);
  const color = useAppSelector((s) => {
    const color = s.regionalPrompts.present.layers.find((l) => l.id === id)?.color;
    if (color) {
      return rgbaColorToString({ ...color, a: selectedLayer === id ? 1 : 0.35 });
    }
    return 'base.700';
  });
  const onClickCapture = useCallback(() => {
    // Must be capture so that the layer is selected before deleting/resetting/etc
    dispatch(layerSelected(id));
  }, [dispatch, id]);
  return (
    <Flex gap={2} onClickCapture={onClickCapture} bg={color} borderRadius="base" p="1px" ps={3}>
      <Flex flexDir="column" gap={2} w="full" bg="base.850" borderRadius="base" p={2}>
        <Flex gap={2} alignItems="center">
          <LayerColorPicker id={id} />
          <LayerVisibilityToggle id={id} />
          <Spacer />
          <LayerAutoNegativeCombobox layerId={id} />
          <LayerMenu id={id} />
        </Flex>
        <RegionalPromptsPositivePrompt layerId={id} />
        <RegionalPromptsNegativePrompt layerId={id} />
      </Flex>
    </Flex>
  );
});

LayerListItem.displayName = 'LayerListItem';
