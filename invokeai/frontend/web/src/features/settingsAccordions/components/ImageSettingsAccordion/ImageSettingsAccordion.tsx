import type { FormLabelProps } from '@invoke-ai/ui-library';
import { Expander, Flex, FormControlGroup, StandaloneAccordion } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectIsFLUX, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice, selectScaleMethod } from 'features/controlLayers/store/selectors';
import { ParamOptimizedDenoisingToggle } from 'features/parameters/components/Advanced/ParamOptimizedDenoisingToggle';
import BboxScaledHeight from 'features/parameters/components/Bbox/BboxScaledHeight';
import BboxScaledWidth from 'features/parameters/components/Bbox/BboxScaledWidth';
import BboxScaleMethod from 'features/parameters/components/Bbox/BboxScaleMethod';
import { BboxSettings } from 'features/parameters/components/Bbox/BboxSettings';
import { ParamDenoisingStrength } from 'features/parameters/components/Core/ParamDenoisingStrength';
import { ParamSeedNumberInput } from 'features/parameters/components/Seed/ParamSeedNumberInput';
import { ParamSeedRandomize } from 'features/parameters/components/Seed/ParamSeedRandomize';
import { ParamSeedShuffle } from 'features/parameters/components/Seed/ParamSeedShuffle';
import { useExpanderToggle } from 'features/settingsAccordions/hooks/useExpanderToggle';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selectBadges = createMemoizedSelector([selectCanvasSlice, selectParamsSlice], (canvas, params) => {
  const { shouldRandomizeSeed } = params;
  const badges: string[] = [];

  const { aspectRatio } = canvas.bbox;
  const { width, height } = canvas.bbox.rect;

  badges.push(`${width}×${height}`);
  badges.push(aspectRatio.id);

  if (aspectRatio.isLocked) {
    badges.push('locked');
  }

  if (!shouldRandomizeSeed) {
    badges.push('Manual Seed');
  }

  if (badges.length === 0) {
    return EMPTY_ARRAY;
  }

  badges;
});

const scalingLabelProps: FormLabelProps = {
  minW: '4.5rem',
};

export const ImageSettingsAccordion = memo(() => {
  const { t } = useTranslation();
  const badges = useAppSelector(selectBadges);
  const scaleMethod = useAppSelector(selectScaleMethod);
  const { isOpen: isOpenAccordion, onToggle: onToggleAccordion } = useStandaloneAccordionToggle({
    id: 'image-settings',
    defaultIsOpen: true,
  });
  const { isOpen: isOpenExpander, onToggle: onToggleExpander } = useExpanderToggle({
    id: 'image-settings-advanced',
    defaultIsOpen: false,
  });
  const isFLUX = useAppSelector(selectIsFLUX);

  return (
    <StandaloneAccordion
      label={t('accordions.image.title')}
      badges={badges}
      isOpen={isOpenAccordion}
      onToggle={onToggleAccordion}
    >
      <Flex px={4} pt={4} w="full" h="full" flexDir="column" data-testid="image-settings-accordion">
        <BboxSettings />
        <Flex py={3} gap={4} alignItems="center">
          <ParamSeedNumberInput />
          <ParamSeedShuffle />
          <ParamSeedRandomize />
        </Flex>
        <ParamDenoisingStrength />
        <Expander label={t('accordions.advanced.options')} isOpen={isOpenExpander} onToggle={onToggleExpander}>
          <Flex gap={4} pb={4} flexDir="column">
            {isFLUX && <ParamOptimizedDenoisingToggle />}
            <BboxScaleMethod />
            {scaleMethod !== 'none' && (
              <FormControlGroup formLabelProps={scalingLabelProps}>
                <BboxScaledWidth />
                <BboxScaledHeight />
              </FormControlGroup>
            )}
          </Flex>
        </Expander>
      </Flex>
    </StandaloneAccordion>
  );
});

ImageSettingsAccordion.displayName = 'ImageSettingsAccordion';
