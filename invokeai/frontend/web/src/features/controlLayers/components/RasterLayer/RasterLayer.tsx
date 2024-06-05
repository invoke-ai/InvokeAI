import { Flex, Spacer, useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { LayerDeleteButton } from 'features/controlLayers/components/LayerCommon/LayerDeleteButton';
import { LayerMenu } from 'features/controlLayers/components/LayerCommon/LayerMenu';
import { LayerOpacity } from 'features/controlLayers/components/LayerCommon/LayerOpacity';
import { LayerTitle } from 'features/controlLayers/components/LayerCommon/LayerTitle';
import { LayerIsEnabledToggle } from 'features/controlLayers/components/LayerCommon/LayerVisibilityToggle';
import { LayerWrapper } from 'features/controlLayers/components/LayerCommon/LayerWrapper';
import { layerSelected, selectLayerOrThrow } from 'features/controlLayers/store/controlLayersSlice';
import { isRasterLayer } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';

type Props = {
  layerId: string;
};

export const RasterLayer = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const isSelected = useAppSelector(
    (s) => selectLayerOrThrow(s.controlLayers.present, layerId, isRasterLayer).isSelected
  );
  const onClick = useCallback(() => {
    dispatch(layerSelected(layerId));
  }, [dispatch, layerId]);
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });

  return (
    <LayerWrapper onClick={onClick} borderColor={isSelected ? 'base.400' : 'base.800'}>
      <Flex gap={3} alignItems="center" p={3} cursor="pointer" onDoubleClick={onToggle}>
        <LayerIsEnabledToggle layerId={layerId} />
        <LayerTitle type="raster_layer" />
        <Spacer />
        <LayerOpacity layerId={layerId} />
        <LayerMenu layerId={layerId} />
        <LayerDeleteButton layerId={layerId} />
      </Flex>
      {isOpen && (
        <Flex flexDir="column" gap={3} px={3} pb={3}>
          PLACEHOLDER
        </Flex>
      )}
    </LayerWrapper>
  );
});

RasterLayer.displayName = 'RasterLayer';
