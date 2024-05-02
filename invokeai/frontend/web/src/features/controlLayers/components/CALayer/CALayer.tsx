import { Flex, Spacer, useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { CALayerControlAdapterWrapper } from 'features/controlLayers/components/CALayer/CALayerControlAdapterWrapper';
import { LayerDeleteButton } from 'features/controlLayers/components/LayerCommon/LayerDeleteButton';
import { LayerMenu } from 'features/controlLayers/components/LayerCommon/LayerMenu';
import { LayerTitle } from 'features/controlLayers/components/LayerCommon/LayerTitle';
import { LayerVisibilityToggle } from 'features/controlLayers/components/LayerCommon/LayerVisibilityToggle';
import { LayerWrapper } from 'features/controlLayers/components/LayerCommon/LayerWrapper';
import { layerSelected, selectCALayerOrThrow } from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback } from 'react';

import CALayerOpacity from './CALayerOpacity';

type Props = {
  layerId: string;
};

export const CALayer = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const isSelected = useAppSelector((s) => selectCALayerOrThrow(s.controlLayers.present, layerId).isSelected);
  const onClick = useCallback(() => {
    // Must be capture so that the layer is selected before deleting/resetting/etc
    dispatch(layerSelected(layerId));
  }, [dispatch, layerId]);
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });

  return (
    <LayerWrapper onClick={onClick} borderColor={isSelected ? 'base.400' : 'base.800'}>
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
    </LayerWrapper>
  );
});

CALayer.displayName = 'CALayer';
