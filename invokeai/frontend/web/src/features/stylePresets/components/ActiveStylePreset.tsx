import { Flex, IconButton, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { negativePromptChanged, positivePromptChanged } from 'features/controlLayers/store/controlLayersSlice';
import { usePresetModifiedPrompts } from 'features/stylePresets/hooks/usePresetModifiedPrompts';
import { activeStylePresetChanged, viewModeChanged } from 'features/stylePresets/store/stylePresetSlice';
import type { MouseEventHandler } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEyeBold, PiStackSimpleBold, PiXBold } from 'react-icons/pi';

import StylePresetImage from './StylePresetImage';

export const ActiveStylePreset = () => {
  const { activeStylePreset, viewMode } = useAppSelector((s) => s.stylePreset);

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

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
          {t('stylePresets.choosePromptTemplate')}
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
          <Tooltip label={t('stylePresets.toggleViewMode')}>
            <IconButton
              onClick={handleToggleViewMode}
              variant="outline"
              size="sm"
              aria-label={t('stylePresets.toggleViewMode')}
              colorScheme={viewMode ? 'invokeBlue' : 'base'}
              icon={<PiEyeBold />}
            />
          </Tooltip>
          <Tooltip label={t('stylePresets.flatten')}>
            <IconButton
              onClick={handleFlattenPrompts}
              variant="outline"
              size="sm"
              aria-label={t('stylePresets.flatten')}
              icon={<PiStackSimpleBold />}
            />
          </Tooltip>
          <Tooltip label={t('stylePresets.clearTemplateSelection')}>
            <IconButton
              onClick={handleClearActiveStylePreset}
              variant="outline"
              size="sm"
              aria-label={t('stylePresets.clearTemplateSelection')}
              icon={<PiXBold />}
            />
          </Tooltip>
        </Flex>
      </Flex>
    </>
  );
};
