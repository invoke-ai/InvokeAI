import { Flex, Spacer, useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { LayerDeleteButton } from 'features/controlLayers/components/LayerCommon/LayerDeleteButton';
import { LayerMenu } from 'features/controlLayers/components/LayerCommon/LayerMenu';
import { LayerTitle } from 'features/controlLayers/components/LayerCommon/LayerTitle';
import { LayerIsEnabledToggle } from 'features/controlLayers/components/LayerCommon/LayerVisibilityToggle';
import { LayerWrapper } from 'features/controlLayers/components/LayerCommon/LayerWrapper';
import { layerSelected, selectRasterLayerOrThrow } from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback } from 'react';

import { RasterLayerOpacity } from './RasterLayerOpacity';

type Props = {
  layerId: string;
};

export const RasterLayer = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const isSelected = useAppSelector((s) => selectRasterLayerOrThrow(s.controlLayers.present, layerId).isSelected);
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
        <RasterLayerOpacity layerId={layerId} />
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
