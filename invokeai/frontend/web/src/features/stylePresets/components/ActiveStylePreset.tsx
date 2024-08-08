import { Flex, IconButton, Text, Box, ButtonGroup } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { negativePromptChanged, positivePromptChanged } from 'features/controlLayers/store/controlLayersSlice';
import { usePresetModifiedPrompts } from 'features/stylePresets/hooks/usePresetModifiedPrompts';
import { activeStylePresetChanged, viewModeChanged } from 'features/stylePresets/store/stylePresetSlice';
import type { MouseEventHandler } from 'react';
import { useCallback } from 'react';
import { PiEyeBold, PiStackSimpleBold, PiXBold } from 'react-icons/pi';
import StylePresetImage from './StylePresetImage';

export const ActiveStylePreset = () => {
  const { activeStylePreset, viewMode } = useAppSelector((s) => s.stylePreset);

  const dispatch = useAppDispatch();

  const { presetModifiedPositivePrompt, presetModifiedNegativePrompt } = usePresetModifiedPrompts();

  const handleClearActiveStylePreset = useCallback<MouseEventHandler<HTMLButtonElement>>(
    (e) => {
      e.stopPropagation();
      dispatch(viewModeChanged(false));
      dispatch(activeStylePresetChanged(null));
    },
    [dispatch]
  );

  const handleFlattenPrompts = useCallback<MouseEventHandler<HTMLButtonElement>>(
    (e) => {
      e.stopPropagation();
      dispatch(positivePromptChanged(presetModifiedPositivePrompt));
      dispatch(negativePromptChanged(presetModifiedNegativePrompt));
      dispatch(viewModeChanged(false));
      dispatch(activeStylePresetChanged(null));
    },
    [dispatch, presetModifiedPositivePrompt, presetModifiedNegativePrompt]
  );

  const handleToggleViewMode = useCallback<MouseEventHandler<HTMLButtonElement>>(
    (e) => {
      e.stopPropagation();
      dispatch(viewModeChanged(!viewMode));
    },
    [dispatch, viewMode]
  );

  if (!activeStylePreset) {
    return (
      <Flex h="25px" alignItems="center">
        <Text fontSize="sm" fontWeight="semibold" color="base.300">
          Choose Preset
        </Text>
      </Flex>
    );
  }
  return (
    <>
      <Flex justifyContent="space-between" w="full" alignItems="center">
        <Flex gap="2" alignItems="center">
          <StylePresetImage imageWidth={25} presetImageUrl={activeStylePreset.image} />
          <Flex flexDir="column">
            <Text fontSize="sm" fontWeight="semibold" color="base.300" noOfLines={1}>
              {activeStylePreset.name}
            </Text>
          </Flex>
        </Flex>
        <Flex gap="1">
          <IconButton
            onClick={handleToggleViewMode}
            variant="outline"
            size="sm"
            aria-label="View"
            colorScheme={viewMode ? 'invokeBlue' : 'base'}
            icon={<PiEyeBold />}
          />
          <IconButton
            onClick={handleFlattenPrompts}
            variant="outline"
            size="sm"
            aria-label="Flatten"
            icon={<PiStackSimpleBold />}
          />
          <IconButton
            onClick={handleClearActiveStylePreset}
            variant="outline"
            size="sm"
            aria-label="Clear"
            icon={<PiXBold />}
          />
        </Flex>
      </Flex>
    </>
  );
};
