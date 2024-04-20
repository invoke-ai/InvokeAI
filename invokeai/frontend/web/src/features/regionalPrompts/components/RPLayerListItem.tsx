import { Flex, Spacer } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { rgbColorToString } from 'features/canvas/util/colorToString';
import { RPLayerActionsButtonGroup } from 'features/regionalPrompts/components/RPLayerActionsButtonGroup';
import { RPLayerAutoNegativeCombobox } from 'features/regionalPrompts/components/RPLayerAutoNegativeCombobox';
import { RPLayerColorPicker } from 'features/regionalPrompts/components/RPLayerColorPicker';
import { RPLayerMenu } from 'features/regionalPrompts/components/RPLayerMenu';
import { RPLayerNegativePrompt } from 'features/regionalPrompts/components/RPLayerNegativePrompt';
import { RPLayerPositivePrompt } from 'features/regionalPrompts/components/RPLayerPositivePrompt';
import { RPLayerVisibilityToggle } from 'features/regionalPrompts/components/RPLayerVisibilityToggle';
import { isVectorMaskLayer, layerSelected } from 'features/regionalPrompts/store/regionalPromptsSlice';
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
    assert(isVectorMaskLayer(layer), `Layer ${layerId} not found or not an RP layer`);
    return rgbColorToString(layer.previewColor);
  });
  const hasTextPrompt = useAppSelector((s) => {
    const layer = s.regionalPrompts.present.layers.find((l) => l.id === layerId);
    assert(isVectorMaskLayer(layer), `Layer ${layerId} not found or not an RP layer`);
    return layer.textPrompt !== null;
  });
  const onClickCapture = useCallback(() => {
    // Must be capture so that the layer is selected before deleting/resetting/etc
    dispatch(layerSelected(layerId));
  }, [dispatch, layerId]);
  return (
    <Flex
      gap={2}
      onClickCapture={onClickCapture}
      bg={color}
      px={2}
      borderRadius="base"
      borderTop="1px"
      borderBottom="1px"
      borderColor="base.800"
      opacity={selectedLayerId === layerId ? 1 : 0.5}
      cursor="pointer"
    >
      <Flex flexDir="column" gap={2} w="full" bg="base.850" p={2}>
        <Flex gap={2} alignItems="center">
          <RPLayerMenu layerId={layerId} />
          <RPLayerColorPicker layerId={layerId} />
          <RPLayerVisibilityToggle layerId={layerId} />
          <Spacer />
          <RPLayerAutoNegativeCombobox layerId={layerId} />
          <RPLayerActionsButtonGroup layerId={layerId} />
        </Flex>
        {hasTextPrompt && <RPLayerPositivePrompt layerId={layerId} />}
        {hasTextPrompt && <RPLayerNegativePrompt layerId={layerId} />}
      </Flex>
    </Flex>
  );
});

RPLayerListItem.displayName = 'RPLayerListItem';
