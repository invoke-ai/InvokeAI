import { Flex, Spacer, useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IPALayerIPAdapterWrapper } from 'features/controlLayers/components/IPALayer/IPALayerIPAdapterWrapper';
import { LayerDeleteButton } from 'features/controlLayers/components/LayerCommon/LayerDeleteButton';
import { LayerTitle } from 'features/controlLayers/components/LayerCommon/LayerTitle';
import { LayerIsEnabledToggle } from 'features/controlLayers/components/LayerCommon/LayerVisibilityToggle';
import { LayerWrapper } from 'features/controlLayers/components/LayerCommon/LayerWrapper';
import { layerSelected, selectLayerOrThrow } from 'features/controlLayers/store/controlLayersSlice';
import { isIPAdapterLayer } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';

type Props = {
  layerId: string;
};

export const IPALayer = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const isSelected = useAppSelector(
    (s) => selectLayerOrThrow(s.canvasV2, layerId, isIPAdapterLayer).isSelected
  );
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });
  const onClick = useCallback(() => {
    dispatch(layerSelected(layerId));
  }, [dispatch, layerId]);
  return (
    <LayerWrapper onClick={onClick} borderColor={isSelected ? 'base.400' : 'base.800'}>
      <Flex gap={3} alignItems="center" p={3} cursor="pointer" onDoubleClick={onToggle}>
        <LayerIsEnabledToggle layerId={layerId} />
        <LayerTitle type="ip_adapter_layer" />
        <Spacer />
        <LayerDeleteButton layerId={layerId} />
      </Flex>
      {isOpen && (
        <Flex flexDir="column" gap={3} px={3} pb={3}>
          <IPALayerIPAdapterWrapper layerId={layerId} />
        </Flex>
      )}
    </LayerWrapper>
  );
});

IPALayer.displayName = 'IPALayer';
