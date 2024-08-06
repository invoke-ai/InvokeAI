import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { negativePromptChanged, positivePromptChanged } from 'features/controlLayers/store/controlLayersSlice';
import ModelImage from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelImage';
import { usePresetModifiedPrompts } from 'features/stylePresets/hooks/usePresetModifiedPrompts';
import { activeStylePresetChanged } from 'features/stylePresets/store/stylePresetSlice';
import type { MouseEventHandler} from 'react';
import { useCallback } from 'react';
import { CgPushDown } from 'react-icons/cg';
import { PiXBold } from 'react-icons/pi';

export const ActiveStylePreset = () => {
  const { activeStylePreset } = useAppSelector((s) => s.stylePreset);
  const dispatch = useAppDispatch();

  const { presetModifiedPositivePrompt, presetModifiedNegativePrompt } = usePresetModifiedPrompts();

  const handleClearActiveStylePreset = useCallback<MouseEventHandler<HTMLButtonElement>>(
    (e) => {
      e.stopPropagation();
      dispatch(activeStylePresetChanged(null));
    },
    [dispatch]
  );

  const handleFlattenPrompts = useCallback<MouseEventHandler<HTMLButtonElement>>(
    (e) => {
      e.stopPropagation();
      dispatch(positivePromptChanged(presetModifiedPositivePrompt));
      dispatch(negativePromptChanged(presetModifiedNegativePrompt));
      dispatch(activeStylePresetChanged(null));
    },
    [dispatch, presetModifiedPositivePrompt, presetModifiedNegativePrompt]
  );

  if (!activeStylePreset) {
    return <>Choose Preset</>;
  }
  return (
    <>
      <Flex justifyContent="space-between" w="full" alignItems="center">
        <Flex gap="2">
          <ModelImage image_url={null} />
          <Flex flexDir="column">
            <Text variant="subtext" fontSize="xs">
              Prompt Style
            </Text>
            <Text fontSize="md" fontWeight="semibold">
              {activeStylePreset.name}
            </Text>
          </Flex>
        </Flex>
        <Flex gap="1">
          <IconButton
            onClick={handleFlattenPrompts}
            variant="ghost"
            size="md"
            aria-label="Flatten"
            icon={<CgPushDown />}
          />
          <IconButton
            onClick={handleClearActiveStylePreset}
            variant="ghost"
            size="md"
            aria-label="Clear"
            icon={<PiXBold />}
          />
        </Flex>
      </Flex>
    </>
  );
};
