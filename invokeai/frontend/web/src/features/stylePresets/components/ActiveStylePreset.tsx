import { Badge, Flex, IconButton, Spacer, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { negativePromptChanged, positivePromptChanged } from 'features/controlLayers/store/paramsSlice';
import { usePresetModifiedPrompts } from 'features/stylePresets/hooks/usePresetModifiedPrompts';
import {
  activeStylePresetIdChanged,
  selectStylePresetActivePresetId,
  selectStylePresetViewMode,
  viewModeChanged,
} from 'features/stylePresets/store/stylePresetSlice';
import type { MouseEventHandler } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEyeBold, PiStackSimpleBold, PiXBold } from 'react-icons/pi';
import { useListStylePresetsQuery } from 'services/api/endpoints/stylePresets';

import StylePresetImage from './StylePresetImage';

export const ActiveStylePreset = () => {
  const viewMode = useAppSelector(selectStylePresetViewMode);
  const activeStylePresetId = useAppSelector(selectStylePresetActivePresetId);

  const { activeStylePreset } = useListStylePresetsQuery(undefined, {
    selectFromResult: ({ data }) => {
      let activeStylePreset = null;
      if (data) {
        activeStylePreset = data.find((sp) => sp.id === activeStylePresetId);
      }
      return { activeStylePreset };
    },
  });

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const { presetModifiedPositivePrompt, presetModifiedNegativePrompt } = usePresetModifiedPrompts();

  const handleClearActiveStylePreset = useCallback<MouseEventHandler<HTMLButtonElement>>(
    (e) => {
      e.stopPropagation();
      dispatch(viewModeChanged(false));
      dispatch(activeStylePresetIdChanged(null));
    },
    [dispatch]
  );

  const handleFlattenPrompts = useCallback<MouseEventHandler<HTMLButtonElement>>(
    (e) => {
      e.stopPropagation();
      dispatch(positivePromptChanged(presetModifiedPositivePrompt));
      dispatch(negativePromptChanged(presetModifiedNegativePrompt));
      dispatch(viewModeChanged(false));
      dispatch(activeStylePresetIdChanged(null));
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
      <Flex h={25} alignItems="center">
        <Text fontSize="sm" fontWeight="semibold" color="base.300">
          {t('stylePresets.choosePromptTemplate')}
        </Text>
      </Flex>
    );
  }
  return (
    <Flex w="full" alignItems="center" gap={2} minW={0}>
      <StylePresetImage imageWidth={25} presetImageUrl={activeStylePreset.image} />
      <Badge colorScheme="invokeBlue" variant="subtle" justifySelf="flex-start">
        {activeStylePreset.name}
      </Badge>
      <Spacer />
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
  );
};
