import type { FormLabelProps } from '@invoke-ai/ui-library';
import { Expander, Flex, FormControlGroup, StandaloneAccordion } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectModelSupportsAspectRatio,
  selectModelSupportsOptimizedDenoising,
  selectModelSupportsPixelDimensions,
  selectModelSupportsSeed,
  selectShouldRandomizeSeed,
} from 'features/controlLayers/store/paramsSlice';
import { selectBbox, selectScaleMethod } from 'features/controlLayers/store/selectors';
import { ParamOptimizedDenoisingToggle } from 'features/parameters/components/Advanced/ParamOptimizedDenoisingToggle';
import BboxScaledHeight from 'features/parameters/components/Bbox/BboxScaledHeight';
import BboxScaledWidth from 'features/parameters/components/Bbox/BboxScaledWidth';
import BboxScaleMethod from 'features/parameters/components/Bbox/BboxScaleMethod';
import { BboxSettings } from 'features/parameters/components/Bbox/BboxSettings';
import { ParamSeed } from 'features/parameters/components/Seed/ParamSeed';
import { useExpanderToggle } from 'features/settingsAccordions/hooks/useExpanderToggle';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selectBadges = createMemoizedSelector(
  [
    selectBbox,
    selectShouldRandomizeSeed,
    selectModelSupportsSeed,
    selectModelSupportsAspectRatio,
    selectModelSupportsPixelDimensions,
  ],
  (bbox, shouldRandomizeSeed, modelSupportsSeed, modelSupportsAspectRatio, modelSupportsPixelDimensions) => {
    const badges: string[] = [];

    const { aspectRatio, rect } = bbox;
    const { width, height } = rect;

    if (modelSupportsPixelDimensions) {
      badges.push(`${width}×${height}`);
    }

    if (modelSupportsAspectRatio) {
      badges.push(aspectRatio.id);

      // If a model does not support pixel dimensions, the ratio is essentially always locked.
      if (modelSupportsPixelDimensions && aspectRatio.isLocked) {
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

const scalingLabelProps: FormLabelProps = {
  minW: '4.5rem',
};

export const CanvasTabImageSettingsAccordion = memo(() => {
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
  const modelSupportsOptimizedDenoising = useAppSelector(selectModelSupportsOptimizedDenoising);
  const modelSupportsSeed = useAppSelector(selectModelSupportsSeed);
  const modelSupportsAspectRatio = useAppSelector(selectModelSupportsAspectRatio);
  const modelSupportsPixelDimensions = useAppSelector(selectModelSupportsPixelDimensions);

  if (!modelSupportsAspectRatio && !modelSupportsSeed) {
    return null;
  }

  const withAdvancedSettingsExpander = modelSupportsPixelDimensions;

  return (
    <StandaloneAccordion
      label={t('accordions.image.title')}
      badges={badges}
      isOpen={isOpenAccordion}
      onToggle={onToggleAccordion}
    >
      <Flex
        px={4}
        pt={4}
        pb={withAdvancedSettingsExpander ? 0 : 4}
        w="full"
        h="full"
        flexDir="column"
        data-testid="image-settings-accordion"
      >
        <BboxSettings />
        {modelSupportsSeed && <ParamSeed pt={3} pb={withAdvancedSettingsExpander ? 0 : 3} />}
        {withAdvancedSettingsExpander && (
          <Expander label={t('accordions.advanced.options')} isOpen={isOpenExpander} onToggle={onToggleExpander}>
            <Flex gap={4} pb={4} flexDir="column">
              {modelSupportsOptimizedDenoising && <ParamOptimizedDenoisingToggle />}
              <BboxScaleMethod />
              {scaleMethod !== 'none' && (
                <FormControlGroup formLabelProps={scalingLabelProps}>
                  <BboxScaledWidth />
                  <BboxScaledHeight />
                </FormControlGroup>
              )}
            </Flex>
          </Expander>
        )}
      </Flex>
    </StandaloneAccordion>
  );
});

CanvasTabImageSettingsAccordion.displayName = 'CanvasTabImageSettingsAccordion';
