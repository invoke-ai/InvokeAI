import { Flex, Spacer, useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDroppable from 'common/components/IAIDroppable';
import { LayerDeleteButton } from 'features/controlLayers/components/LayerCommon/LayerDeleteButton';
import { EntityMenu } from 'features/controlLayers/components/LayerCommon/LayerMenu';
import { LayerOpacity } from 'features/controlLayers/components/LayerCommon/LayerOpacity';
import { EntityTitle } from 'features/controlLayers/components/LayerCommon/LayerTitle';
import { EntityEnabledToggle } from 'features/controlLayers/components/LayerCommon/LayerVisibilityToggle';
import { LayerWrapper } from 'features/controlLayers/components/LayerCommon/LayerWrapper';
import { layerSelected, selectLayerOrThrow } from 'features/controlLayers/store/controlLayersSlice';
import { isRasterLayer } from 'features/controlLayers/store/types';
import type { LayerImageDropData } from 'features/dnd/types';
import { memo, useCallback, useMemo } from 'react';

type Props = {
  layerId: string;
};

export const RasterLayer = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const isSelected = useAppSelector(
    (s) => selectLayerOrThrow(s.canvasV2, layerId, isRasterLayer).isSelected
  );
  const onClick = useCallback(() => {
    dispatch(layerSelected(layerId));
  }, [dispatch, layerId]);
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });

  const droppableData = useMemo(() => {
    const _droppableData: LayerImageDropData = {
      id: layerId,
      actionType: 'ADD_RASTER_LAYER_IMAGE',
      context: { layerId },
    };
    return _droppableData;
  }, [layerId]);

  return (
    <LayerWrapper onClick={onClick} borderColor={isSelected ? 'base.400' : 'base.800'}>
      <Flex gap={3} alignItems="center" p={3} cursor="pointer" onDoubleClick={onToggle}>
        <EntityEnabledToggle layerId={layerId} />
        <EntityTitle type="raster_layer" />
        <Spacer />
        <LayerOpacity layerId={layerId} />
        <EntityMenu layerId={layerId} />
        <LayerDeleteButton layerId={layerId} />
      </Flex>
      {isOpen && (
        <Flex flexDir="column" gap={3} px={3} pb={3}>
          PLACEHOLDER
        </Flex>
      )}
      <IAIDroppable data={droppableData} />
    </LayerWrapper>
  );
});

RasterLayer.displayName = 'RasterLayer';
