import { Flex, Spacer, useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InitialImagePreview } from 'features/controlLayers/components/IILayer/InitialImagePreview';
import { LayerDeleteButton } from 'features/controlLayers/components/LayerCommon/LayerDeleteButton';
import { LayerMenu } from 'features/controlLayers/components/LayerCommon/LayerMenu';
import { LayerOpacity } from 'features/controlLayers/components/LayerCommon/LayerOpacity';
import { LayerTitle } from 'features/controlLayers/components/LayerCommon/LayerTitle';
import { LayerIsEnabledToggle } from 'features/controlLayers/components/LayerCommon/LayerVisibilityToggle';
import { LayerWrapper } from 'features/controlLayers/components/LayerCommon/LayerWrapper';
import {
  iiLayerDenoisingStrengthChanged,
  iiLayerImageChanged,
  layerSelected,
  selectLayerOrThrow,
} from 'features/controlLayers/store/controlLayersSlice';
import { isInitialImageLayer } from 'features/controlLayers/store/types';
import type { IILayerImageDropData } from 'features/dnd/types';
import ImageToImageStrength from 'features/parameters/components/ImageToImage/ImageToImageStrength';
import { memo, useCallback, useMemo } from 'react';
import type { IILayerImagePostUploadAction, ImageDTO } from 'services/api/types';

type Props = {
  layerId: string;
};

export const IILayer = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const layer = useAppSelector((s) => selectLayerOrThrow(s.controlLayers.present, layerId, isInitialImageLayer));
  const onClick = useCallback(() => {
    dispatch(layerSelected(layerId));
  }, [dispatch, layerId]);
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });

  const onChangeImage = useCallback(
    (imageDTO: ImageDTO | null) => {
      dispatch(iiLayerImageChanged({ layerId, imageDTO }));
    },
    [dispatch, layerId]
  );

  const onChangeDenoisingStrength = useCallback(
    (denoisingStrength: number) => {
      dispatch(iiLayerDenoisingStrengthChanged({ layerId, denoisingStrength }));
    },
    [dispatch, layerId]
  );

  const droppableData = useMemo<IILayerImageDropData>(
    () => ({
      actionType: 'SET_II_LAYER_IMAGE',
      context: {
        layerId,
      },
      id: layerId,
    }),
    [layerId]
  );

  const postUploadAction = useMemo<IILayerImagePostUploadAction>(
    () => ({
      layerId,
      type: 'SET_II_LAYER_IMAGE',
    }),
    [layerId]
  );

  return (
    <LayerWrapper onClick={onClick} borderColor={layer.isSelected ? 'base.400' : 'base.800'}>
      <Flex gap={3} alignItems="center" p={3} cursor="pointer" onDoubleClick={onToggle}>
        <LayerIsEnabledToggle layerId={layerId} />
        <LayerTitle type="initial_image_layer" />
        <Spacer />
        <LayerOpacity layerId={layerId} />
        <LayerMenu layerId={layerId} />
        <LayerDeleteButton layerId={layerId} />
      </Flex>
      {isOpen && (
        <Flex flexDir="column" gap={3} px={3} pb={3}>
          <ImageToImageStrength value={layer.denoisingStrength} onChange={onChangeDenoisingStrength} />
          <InitialImagePreview
            image={layer.image}
            onChangeImage={onChangeImage}
            droppableData={droppableData}
            postUploadAction={postUploadAction}
          />
        </Flex>
      )}
    </LayerWrapper>
  );
});

IILayer.displayName = 'IILayer';
