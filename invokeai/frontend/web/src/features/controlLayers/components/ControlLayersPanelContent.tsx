/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { AddLayerButton } from 'features/controlLayers/components/AddLayerButton';
import { CA } from 'features/controlLayers/components/ControlAdapter/CA';
import { DeleteAllLayersButton } from 'features/controlLayers/components/DeleteAllLayersButton';
import { IPA } from 'features/controlLayers/components/IPAdapter/IPA';
import { Layer } from 'features/controlLayers/components/Layer/Layer';
import { RG } from 'features/controlLayers/components/RegionalGuidance/RG';
import { mapId } from 'features/controlLayers/konva/util';
import { selectControlAdaptersV2Slice } from 'features/controlLayers/store/controlAdaptersSlice';
import { selectIPAdaptersSlice } from 'features/controlLayers/store/ipAdaptersSlice';
import { selectLayersSlice } from 'features/controlLayers/store/layersSlice';
import { selectRegionalGuidanceSlice } from 'features/controlLayers/store/regionalGuidanceSlice';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const selectRGIds = createMemoizedSelector(selectRegionalGuidanceSlice, (rgState) => {
  return rgState.regions.map(mapId).reverse();
});

const selectCAIds = createMemoizedSelector(selectControlAdaptersV2Slice, (caState) => {
  return caState.controlAdapters.map(mapId).reverse();
});

const selectIPAIds = createMemoizedSelector(selectIPAdaptersSlice, (ipaState) => {
  return ipaState.ipAdapters.map(mapId).reverse();
});

const selectLayerIds = createMemoizedSelector(selectLayersSlice, (layersState) => {
  return layersState.layers.map(mapId).reverse();
});

export const ControlLayersPanelContent = memo(() => {
  const { t } = useTranslation();
  const rgIds = useAppSelector(selectRGIds);
  const caIds = useAppSelector(selectCAIds);
  const ipaIds = useAppSelector(selectIPAIds);
  const layerIds = useAppSelector(selectLayerIds);
  const entityCount = useMemo(
    () => rgIds.length + caIds.length + ipaIds.length + layerIds.length,
    [rgIds.length, caIds.length, ipaIds.length, layerIds.length]
  );

  return (
    <Flex flexDir="column" gap={2} w="full" h="full">
      <Flex justifyContent="space-around">
        <AddLayerButton />
        <DeleteAllLayersButton />
      </Flex>
      {entityCount > 0 && (
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
      )}
      {entityCount === 0 && <IAINoContentFallback icon={null} label={t('controlLayers.noLayersAdded')} />}
    </Flex>
  );
});

ControlLayersPanelContent.displayName = 'ControlLayersPanelContent';
