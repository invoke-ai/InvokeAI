import { Badge, Flex, Spacer, useDisclosure } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { rgbColorToString } from 'features/canvas/util/colorToString';
import { AddPromptButtons } from 'features/controlLayers/components/AddPromptButtons';
import { LayerDeleteButton } from 'features/controlLayers/components/LayerCommon/LayerDeleteButton';
import { LayerMenu } from 'features/controlLayers/components/LayerCommon/LayerMenu';
import { LayerTitle } from 'features/controlLayers/components/LayerCommon/LayerTitle';
import { LayerVisibilityToggle } from 'features/controlLayers/components/LayerCommon/LayerVisibilityToggle';
import {
  isRegionalGuidanceLayer,
  layerSelected,
  selectControlLayersSlice,
} from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { assert } from 'tsafe';

import { RGLayerColorPicker } from './RGLayerColorPicker';
import { RGLayerIPAdapterList } from './RGLayerIPAdapterList';
import { RGLayerNegativePrompt } from './RGLayerNegativePrompt';
import { RGLayerPositivePrompt } from './RGLayerPositivePrompt';
import RGLayerSettingsPopover from './RGLayerSettingsPopover';

type Props = {
  layerId: string;
};

export const RGLayer = memo(({ layerId }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectControlLayersSlice, (controlLayers) => {
        const layer = controlLayers.present.layers.find((l) => l.id === layerId);
        assert(isRegionalGuidanceLayer(layer), `Layer ${layerId} not found or not an RP layer`);
        return {
          color: rgbColorToString(layer.previewColor),
          hasPositivePrompt: layer.positivePrompt !== null,
          hasNegativePrompt: layer.negativePrompt !== null,
          hasIPAdapters: layer.ipAdapterIds.length > 0,
          isSelected: layerId === controlLayers.present.selectedLayerId,
          autoNegative: layer.autoNegative,
        };
      }),
    [layerId]
  );
  const { autoNegative, color, hasPositivePrompt, hasNegativePrompt, hasIPAdapters, isSelected } =
    useAppSelector(selector);
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });
  const onClick = useCallback(() => {
    dispatch(layerSelected(layerId));
  }, [dispatch, layerId]);
  return (
    <Flex gap={2} onClick={onClick} bg={isSelected ? color : 'base.800'} px={2} borderRadius="base" py="1px">
      <Flex flexDir="column" w="full" bg="base.850" borderRadius="base">
        <Flex gap={3} alignItems="center" p={3} cursor="pointer" onDoubleClick={onToggle}>
          <LayerVisibilityToggle layerId={layerId} />
          <LayerTitle type="regional_guidance_layer" />
          <Spacer />
          {autoNegative === 'invert' && (
            <Badge color="base.300" bg="transparent" borderWidth={1} userSelect="none">
              {t('controlLayers.autoNegative')}
            </Badge>
          )}
          <RGLayerColorPicker layerId={layerId} />
          <RGLayerSettingsPopover layerId={layerId} />
          <LayerMenu layerId={layerId} />
          <LayerDeleteButton layerId={layerId} />
        </Flex>
        {isOpen && (
          <Flex flexDir="column" gap={3} px={3} pb={3}>
            {!hasPositivePrompt && !hasNegativePrompt && !hasIPAdapters && <AddPromptButtons layerId={layerId} />}
            {hasPositivePrompt && <RGLayerPositivePrompt layerId={layerId} />}
            {hasNegativePrompt && <RGLayerNegativePrompt layerId={layerId} />}
            {hasIPAdapters && <RGLayerIPAdapterList layerId={layerId} />}
          </Flex>
        )}
      </Flex>
    </Flex>
  );
});

RGLayer.displayName = 'RGLayer';
