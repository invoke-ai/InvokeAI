import { Flex, Spacer } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import ControlAdapterLayerConfig from 'features/regionalPrompts/components/controlAdapterOverrides/ControlAdapterLayerConfig';
import { LayerTitle } from 'features/regionalPrompts/components/LayerTitle';
import { RPLayerDeleteButton } from 'features/regionalPrompts/components/RPLayerDeleteButton';
import { RPLayerMenu } from 'features/regionalPrompts/components/RPLayerMenu';
import { RPLayerVisibilityToggle } from 'features/regionalPrompts/components/RPLayerVisibilityToggle';
import {
  isIPAdapterLayer,
  layerSelected,
  selectRegionalPromptsSlice,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback, useMemo } from 'react';
import { assert } from 'tsafe';

type Props = {
  layerId: string;
};

export const IPAdapterLayerListItem = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
        const layer = regionalPrompts.present.layers.find((l) => l.id === layerId);
        assert(isIPAdapterLayer(layer), `Layer ${layerId} not found or not an IP Adapter layer`);
        return {
          ipAdapterId: layer.ipAdapterId,
          isSelected: layerId === regionalPrompts.present.selectedLayerId,
        };
      }),
    [layerId]
  );
  const { ipAdapterId, isSelected } = useAppSelector(selector);
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
          <LayerTitle type="ip_adapter_layer" />
          <Spacer />
          <RPLayerMenu layerId={layerId} />
          <RPLayerDeleteButton layerId={layerId} />
        </Flex>
        <ControlAdapterLayerConfig id={ipAdapterId} />
      </Flex>
    </Flex>
  );
});

IPAdapterLayerListItem.displayName = 'IPAdapterLayerListItem';
