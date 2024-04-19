import { Flex, Spacer } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { rgbaColorToString } from 'features/canvas/util/colorToString';
import { RPLayerAutoNegativeCombobox } from 'features/regionalPrompts/components/RPLayerAutoNegativeCombobox';
import { RPLayerColorPicker } from 'features/regionalPrompts/components/RPLayerColorPicker';
import { RPLayerMenu } from 'features/regionalPrompts/components/RPLayerMenu';
import { RPLayerNegativePrompt } from 'features/regionalPrompts/components/RPLayerNegativePrompt';
import { RPLayerPositivePrompt } from 'features/regionalPrompts/components/RPLayerPositivePrompt';
import { RPLayerVisibilityToggle } from 'features/regionalPrompts/components/RPLayerVisibilityToggle';
import { isRPLayer, rpLayerSelected } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback } from 'react';
import { assert } from 'tsafe';

type Props = {
  layerId: string;
};

export const RPLayerListItem = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const selectedLayerId = useAppSelector((s) => s.regionalPrompts.present.selectedLayerId);
  const color = useAppSelector((s) => {
    const layer = s.regionalPrompts.present.layers.find((l) => l.id === layerId);
    assert(isRPLayer(layer), `Layer ${layerId} not found or not an RP layer`);
    return rgbaColorToString({ ...layer.color, a: selectedLayerId === layerId ? 1 : 0.35 });
  });
  const onClickCapture = useCallback(() => {
    // Must be capture so that the layer is selected before deleting/resetting/etc
    dispatch(rpLayerSelected(layerId));
  }, [dispatch, layerId]);
  return (
    <Flex gap={2} onClickCapture={onClickCapture} bg={color} borderRadius="base" p="1px" ps={3}>
      <Flex flexDir="column" gap={2} w="full" bg="base.850" borderRadius="base" p={2}>
        <Flex gap={2} alignItems="center">
          <RPLayerColorPicker layerId={layerId} />
          <RPLayerVisibilityToggle layerId={layerId} />
          <Spacer />
          <RPLayerAutoNegativeCombobox layerId={layerId} />
          <RPLayerMenu layerId={layerId} />
        </Flex>
        <RPLayerPositivePrompt layerId={layerId} />
        <RPLayerNegativePrompt layerId={layerId} />
      </Flex>
    </Flex>
  );
});

RPLayerListItem.displayName = 'RPLayerListItem';
