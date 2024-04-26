import { Badge, Flex, Spacer } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { rgbColorToString } from 'features/canvas/util/colorToString';
import { RPLayerColorPicker } from 'features/regionalPrompts/components/RPLayerColorPicker';
import { RPLayerDeleteButton } from 'features/regionalPrompts/components/RPLayerDeleteButton';
import { RPLayerIPAdapterList } from 'features/regionalPrompts/components/RPLayerIPAdapterList';
import { RPLayerMenu } from 'features/regionalPrompts/components/RPLayerMenu';
import { RPLayerNegativePrompt } from 'features/regionalPrompts/components/RPLayerNegativePrompt';
import { RPLayerPositivePrompt } from 'features/regionalPrompts/components/RPLayerPositivePrompt';
import RPLayerSettingsPopover from 'features/regionalPrompts/components/RPLayerSettingsPopover';
import { RPLayerVisibilityToggle } from 'features/regionalPrompts/components/RPLayerVisibilityToggle';
import {
  isMaskedGuidanceLayer,
  layerSelected,
  selectRegionalPromptsSlice,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { assert } from 'tsafe';

import { AddPromptButtons } from './AddPromptButtons';

type Props = {
  layerId: string;
};

export const RPLayerListItem = memo(({ layerId }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
        const layer = regionalPrompts.present.layers.find((l) => l.id === layerId);
        assert(isMaskedGuidanceLayer(layer), `Layer ${layerId} not found or not an RP layer`);
        return {
          color: rgbColorToString(layer.previewColor),
          hasPositivePrompt: layer.positivePrompt !== null,
          hasNegativePrompt: layer.negativePrompt !== null,
          hasIPAdapters: layer.ipAdapterIds.length > 0,
          isSelected: layerId === regionalPrompts.present.selectedLayerId,
          autoNegative: layer.autoNegative,
        };
      }),
    [layerId]
  );
  const { autoNegative, color, hasPositivePrompt, hasNegativePrompt, hasIPAdapters, isSelected } =
    useAppSelector(selector);
  const onClickCapture = useCallback(() => {
    // Must be capture so that the layer is selected before deleting/resetting/etc
    dispatch(layerSelected(layerId));
  }, [dispatch, layerId]);
  return (
    <Flex
      gap={2}
      onClickCapture={onClickCapture}
      bg={isSelected ? color : 'base.800'}
      ps={2}
      borderRadius="base"
      pe="1px"
      py="1px"
      cursor="pointer"
    >
      <Flex flexDir="column" gap={2} w="full" bg="base.850" p={2} borderRadius="base">
        <Flex gap={3} alignItems="center">
          <RPLayerVisibilityToggle layerId={layerId} />
          <RPLayerColorPicker layerId={layerId} />
          <Spacer />
          {autoNegative === 'invert' && (
            <Badge color="base.300" bg="transparent" borderWidth={1}>
              {t('regionalPrompts.autoNegative')}
            </Badge>
          )}
          <RPLayerDeleteButton layerId={layerId} />
          <RPLayerSettingsPopover layerId={layerId} />
          <RPLayerMenu layerId={layerId} />
        </Flex>
        {!hasPositivePrompt && !hasNegativePrompt && !hasIPAdapters && <AddPromptButtons layerId={layerId} />}
        {hasPositivePrompt && <RPLayerPositivePrompt layerId={layerId} />}
        {hasNegativePrompt && <RPLayerNegativePrompt layerId={layerId} />}
        {hasIPAdapters && <RPLayerIPAdapterList layerId={layerId} />}
      </Flex>
    </Flex>
  );
});

RPLayerListItem.displayName = 'RPLayerListItem';
