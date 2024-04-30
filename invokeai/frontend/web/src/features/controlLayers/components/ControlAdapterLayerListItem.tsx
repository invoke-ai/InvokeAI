import { Flex, Spacer } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import CALayerOpacity from 'features/controlLayers/components/CALayerOpacity';
import ControlAdapterLayerConfig from 'features/controlLayers/components/controlAdapterOverrides/ControlAdapterLayerConfig';
import { LayerTitle } from 'features/controlLayers/components/LayerTitle';
import { RPLayerDeleteButton } from 'features/controlLayers/components/RPLayerDeleteButton';
import { RPLayerMenu } from 'features/controlLayers/components/RPLayerMenu';
import { RPLayerVisibilityToggle } from 'features/controlLayers/components/RPLayerVisibilityToggle';
import {
  isControlAdapterLayer,
  layerSelected,
  selectRegionalPromptsSlice,
} from 'features/controlLayers/store/regionalPromptsSlice';
import { memo, useCallback, useMemo } from 'react';
import { assert } from 'tsafe';

type Props = {
  layerId: string;
};

export const ControlAdapterLayerListItem = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
        const layer = regionalPrompts.present.layers.find((l) => l.id === layerId);
        assert(isControlAdapterLayer(layer), `Layer ${layerId} not found or not a ControlNet layer`);
        return {
          controlNetId: layer.controlNetId,
          isSelected: layerId === regionalPrompts.present.selectedLayerId,
        };
      }),
    [layerId]
  );
  const { controlNetId, isSelected } = useAppSelector(selector);
  const onClickCapture = useCallback(() => {
    // Must be capture so that the layer is selected before deleting/resetting/etc
    dispatch(layerSelected(layerId));
  }, [dispatch, layerId]);
  return (
    <Flex
      gap={2}
      onClickCapture={onClickCapture}
      bg={isSelected ? 'base.400' : 'base.800'}
      px={2}
      borderRadius="base"
      py="1px"
    >
      <Flex flexDir="column" gap={4} w="full" bg="base.850" p={3} borderRadius="base">
        <Flex gap={3} alignItems="center" cursor="pointer">
          <RPLayerVisibilityToggle layerId={layerId} />
          <LayerTitle type="control_adapter_layer" />
          <Spacer />
          <CALayerOpacity layerId={layerId} />
          <RPLayerMenu layerId={layerId} />
          <RPLayerDeleteButton layerId={layerId} />
        </Flex>
        <ControlAdapterLayerConfig id={controlNetId} />
      </Flex>
    </Flex>
  );
});

ControlAdapterLayerListItem.displayName = 'ControlAdapterLayerListItem';
