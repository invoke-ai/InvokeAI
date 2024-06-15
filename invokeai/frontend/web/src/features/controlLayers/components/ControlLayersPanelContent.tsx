/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { AddLayerButton } from 'features/controlLayers/components/AddLayerButton';
import { CALayer } from 'features/controlLayers/components/CALayer/CALayer';
import { DeleteAllLayersButton } from 'features/controlLayers/components/DeleteAllLayersButton';
import { IILayer } from 'features/controlLayers/components/IILayer/IILayer';
import { IPAEntity } from 'features/controlLayers/components/IPALayer/IPALayer';
import { Layer } from 'features/controlLayers/components/RasterLayer/RasterLayer';
import { RGLayer } from 'features/controlLayers/components/RGLayer/RGLayer';
import { selectCanvasV2Slice } from 'features/controlLayers/store/controlLayersSlice';
import type { LayerData } from 'features/controlLayers/store/types';
import { isRenderableLayer } from 'features/controlLayers/store/types';
import { partition } from 'lodash-es';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selectLayerIdTypePairs = createMemoizedSelector(selectCanvasV2Slice, (controlLayers) => {
  const [renderableLayers, ipAdapterLayers] = partition(canvasV2.layers, isRenderableLayer);
  return [...ipAdapterLayers, ...renderableLayers].map((l) => ({ id: l.id, type: l.type })).reverse();
});

export const ControlLayersPanelContent = memo(() => {
  const { t } = useTranslation();
  const layerIdTypePairs = useAppSelector(selectLayerIdTypePairs);
  return (
    <Flex flexDir="column" gap={2} w="full" h="full">
      <Flex justifyContent="space-around">
        <AddLayerButton />
        <DeleteAllLayersButton />
      </Flex>
      {layerIdTypePairs.length > 0 && (
        <ScrollableContent>
          <Flex flexDir="column" gap={2} data-testid="control-layers-layer-list">
            {layerIdTypePairs.map(({ id, type }) => (
              <LayerWrapper key={id} id={id} type={type} />
            ))}
          </Flex>
        </ScrollableContent>
      )}
      {layerIdTypePairs.length === 0 && <IAINoContentFallback icon={null} label={t('controlLayers.noLayersAdded')} />}
    </Flex>
  );
});

ControlLayersPanelContent.displayName = 'ControlLayersPanelContent';

type LayerWrapperProps = {
  id: string;
  type: LayerData['type'];
};

const LayerWrapper = memo(({ id, type }: LayerWrapperProps) => {
  if (type === 'regional_guidance_layer') {
    return <RGLayer key={id} layerId={id} />;
  }
  if (type === 'control_adapter_layer') {
    return <CALayer key={id} id={id} />;
  }
  if (type === 'ip_adapter_layer') {
    return <IPAEntity key={id} layerId={id} />;
  }
  if (type === 'initial_image_layer') {
    return <IILayer key={id} layerId={id} />;
  }
  if (type === 'raster_layer') {
    return <Layer key={id} layerId={id} />;
  }
});

LayerWrapper.displayName = 'LayerWrapper';
