import { Flex, StandaloneAccordion } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectAspectRatioID,
  selectAspectRatioIsLocked,
  selectHeight,
  selectModelSupportsDimensions,
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
  ],
  (width, height, aspectRatioID, aspectRatioIsLocked, shouldRandomizeSeed, modelSupportsSeed) => {
    const badges: string[] = [];

    badges.push(`${width}×${height}`);

    badges.push(aspectRatioID);

    if (aspectRatioIsLocked) {
      badges.push('locked');
    }

    if (modelSupportsSeed && !shouldRandomizeSeed) {
      badges.push('Manual Seed');
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
  const modelSupportsDimensions = useAppSelector(selectModelSupportsDimensions);
  const modelSupportsSeed = useAppSelector(selectModelSupportsSeed);
  const { isOpen: isOpenAccordion, onToggle: onToggleAccordion } = useStandaloneAccordionToggle({
    id: 'image-settings-generate-tab',
    defaultIsOpen: true,
  });

  if (!modelSupportsDimensions && !modelSupportsSeed) {
    return null;
  }

  return (
    <StandaloneAccordion
      label={t('accordions.image.title')}
      badges={badges}
      isOpen={isOpenAccordion}
      onToggle={onToggleAccordion}
    >
      <Flex px={4} pt={4} pb={4} w="full" h="full" flexDir="column" data-testid="image-settings-accordion">
        {modelSupportsDimensions && <Dimensions />}
        {modelSupportsSeed && <ParamSeed py={3} />}
      </Flex>
    </StandaloneAccordion>
  );
});

GenerateTabImageSettingsAccordion.displayName = 'GenerateTabImageSettingsAccordion';
