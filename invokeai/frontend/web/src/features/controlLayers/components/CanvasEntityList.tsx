/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { CA } from 'features/controlLayers/components/ControlAdapter/CA';
import { IPA } from 'features/controlLayers/components/IPAdapter/IPA';
import { Layer } from 'features/controlLayers/components/Layer/Layer';
import { RG } from 'features/controlLayers/components/RegionalGuidance/RG';
import { mapId } from 'features/controlLayers/konva/util';
import { selectCanvasV2Slice } from 'features/controlLayers/store/canvasV2Slice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selectEntityIds = createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => {
  const rgIds = canvasV2.regions.entities.map(mapId).reverse();
  const caIds = canvasV2.controlAdapters.entities.map(mapId).reverse();
  const ipaIds = canvasV2.ipAdapters.entities.map(mapId).reverse();
  const layerIds = canvasV2.layers.entities.map(mapId).reverse();
  const entityCount = rgIds.length + caIds.length + ipaIds.length + layerIds.length;
  return { rgIds, caIds, ipaIds, layerIds, entityCount };
});

export const CanvasEntityList = memo(() => {
  const { t } = useTranslation();
  const { rgIds, caIds, ipaIds, layerIds, entityCount } = useAppSelector(selectEntityIds);

  if (entityCount > 0) {
    return (
      <ScrollableContent>
        <Flex flexDir="column" gap={2} data-testid="control-layers-layer-list">
          {rgIds.map((id) => (
            <RG key={id} id={id} />
          ))}
          {caIds.map((id) => (
            <CA key={id} id={id} />
          ))}
          {ipaIds.map((id) => (
            <IPA key={id} id={id} />
          ))}
          {layerIds.map((id) => (
            <Layer key={id} id={id} />
          ))}
        </Flex>
      </ScrollableContent>
    );
  }

  return <IAINoContentFallback icon={null} label={t('controlLayers.noLayersAdded')} />;
});

CanvasEntityList.displayName = 'CanvasEntityList';
