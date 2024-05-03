import { Flex, Spacer, useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { CALayerControlAdapterWrapper } from 'features/controlLayers/components/CALayer/CALayerControlAdapterWrapper';
import { LayerDeleteButton } from 'features/controlLayers/components/LayerCommon/LayerDeleteButton';
import { LayerMenu } from 'features/controlLayers/components/LayerCommon/LayerMenu';
import { LayerTitle } from 'features/controlLayers/components/LayerCommon/LayerTitle';
import { LayerVisibilityToggle } from 'features/controlLayers/components/LayerCommon/LayerVisibilityToggle';
import { layerSelected, selectCALayerOrThrow } from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback } from 'react';

import CALayerOpacity from './CALayerOpacity';

type Props = {
  layerId: string;
};

export const CALayer = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const isSelected = useAppSelector((s) => selectCALayerOrThrow(s.controlLayers.present, layerId).isSelected);
  const onClickCapture = useCallback(() => {
    // Must be capture so that the layer is selected before deleting/resetting/etc
    dispatch(layerSelected(layerId));
  }, [dispatch, layerId]);
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });

  return (
    <Flex
      gap={2}
      onClickCapture={onClickCapture}
      bg={isSelected ? 'base.400' : 'base.800'}
      px={2}
      borderRadius="base"
      py="1px"
    >
      <Flex flexDir="column" w="full" bg="base.850" borderRadius="base">
        <Flex gap={3} alignItems="center" p={3} cursor="pointer" onDoubleClick={onToggle}>
          <LayerVisibilityToggle layerId={layerId} />
          <LayerTitle type="control_adapter_layer" />
          <Spacer />
          <CALayerOpacity layerId={layerId} />
          <LayerMenu layerId={layerId} />
          <LayerDeleteButton layerId={layerId} />
        </Flex>
        {isOpen && (
          <Flex flexDir="column" gap={3} px={3} pb={3}>
            <CALayerControlAdapterWrapper layerId={layerId} />
          </Flex>
        )}
      </Flex>
    </Flex>
  );
});

CALayer.displayName = 'CALayer';
