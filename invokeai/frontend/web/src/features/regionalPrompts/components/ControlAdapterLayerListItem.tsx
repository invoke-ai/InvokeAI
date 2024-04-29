import { Flex, Spacer } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import ControlAdapterLayerConfig from 'features/regionalPrompts/components/controlAdapterOverrides/ControlAdapterLayerConfig';
import { LayerTitle } from 'features/regionalPrompts/components/LayerTitle';
import { RPLayerDeleteButton } from 'features/regionalPrompts/components/RPLayerDeleteButton';
import { RPLayerVisibilityToggle } from 'features/regionalPrompts/components/RPLayerVisibilityToggle';
import {
  isControlAdapterLayer,
  layerSelected,
  selectRegionalPromptsSlice,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
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
      ps={2}
      borderRadius="base"
      pe="1px"
      py="1px"
    >
      <Flex flexDir="column" gap={4} w="full" bg="base.850" p={3} borderRadius="base">
        <Flex gap={3} alignItems="center">
          <RPLayerVisibilityToggle layerId={layerId} />
          <LayerTitle type="control_adapter_layer" />
          <Spacer />
          <RPLayerDeleteButton layerId={layerId} />
        </Flex>
        <ControlAdapterLayerConfig id={controlNetId} />
      </Flex>
    </Flex>
  );
});

ControlAdapterLayerListItem.displayName = 'ControlAdapterLayerListItem';
