import { Flex, StandaloneAccordion } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectAspectRatioID,
  selectAspectRatioIsLocked,
  selectHeight,
  selectIsApiBaseModel,
  selectModelSupportsAspectRatio,
  selectModelSupportsPixelDimensions,
  selectModelSupportsSeed,
  selectShouldRandomizeSeed,
  selectWidth,
} from 'features/controlLayers/store/paramsSlice';
import { Dimensions } from 'features/parameters/components/Dimensions/Dimensions';
import { ParamSeed } from 'features/parameters/components/Seed/ParamSeed';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selectBadges = createMemoizedSelector(
  [
    selectWidth,
    selectHeight,
    selectAspectRatioID,
    selectAspectRatioIsLocked,
    selectShouldRandomizeSeed,
    selectModelSupportsSeed,
    selectModelSupportsAspectRatio,
    selectModelSupportsPixelDimensions,
  ],
  (
    width,
    height,
    aspectRatioID,
    aspectRatioIsLocked,
    shouldRandomizeSeed,
    modelSupportsSeed,
    modelSupportsAspectRatio,
    modelSupportsPixelDimensions
  ) => {
    const badges: string[] = [];

    if (modelSupportsPixelDimensions) {
      badges.push(`${width}×${height}`);
    }

    if (modelSupportsAspectRatio) {
      badges.push(aspectRatioID);

      // If a model does not support pixel dimensions, the ratio is essentially always locked.
      if (modelSupportsPixelDimensions && aspectRatioIsLocked) {
        badges.push('locked');
      }
    }

    if (modelSupportsSeed) {
      if (!shouldRandomizeSeed) {
        badges.push('Manual Seed');
      }
    }

    if (badges.length === 0) {
      return EMPTY_ARRAY;
    }

    return badges;
  }
);

export const GenerateTabImageSettingsAccordion = memo(() => {
  const { t } = useTranslation();
  const badges = useAppSelector(selectBadges);
  const { isOpen: isOpenAccordion, onToggle: onToggleAccordion } = useStandaloneAccordionToggle({
    id: 'image-settings-generate-tab',
    defaultIsOpen: true,
  });
  const supportsSeed = useAppSelector(selectModelSupportsSeed);
  const supportsAspectRatio = useAppSelector(selectModelSupportsAspectRatio);

  if (!supportsAspectRatio && !supportsSeed) {
    return;
  }

  return (
    <StandaloneAccordion
      label={t('accordions.image.title')}
      badges={badges}
      isOpen={isOpenAccordion}
      onToggle={onToggleAccordion}
    >
      <Flex px={4} pt={4} pb={4} w="full" h="full" flexDir="column" data-testid="image-settings-accordion">
        {supportsAspectRatio && <Dimensions />}
        {supportsSeed && <ParamSeed py={3} />}
      </Flex>
    </StandaloneAccordion>
  );
});

GenerateTabImageSettingsAccordion.displayName = 'GenerateTabImageSettingsAccordion';
