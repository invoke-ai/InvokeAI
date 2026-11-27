import { Expander, Flex, StandaloneAccordion } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { roundDownToMultiple } from 'common/util/roundDownToMultiple';
import ParamCreativity from 'features/parameters/components/Upscale/ParamCreativity';
import ParamSpandrelModel from 'features/parameters/components/Upscale/ParamSpandrelModel';
import ParamStructure from 'features/parameters/components/Upscale/ParamStructure';
import { selectUpscaleSlice } from 'features/parameters/store/upscaleSlice';
import { getGridSize } from 'features/parameters/util/optimalDimension';
import { UpscaleScaleSlider } from 'features/settingsAccordions/components/UpscaleSettingsAccordion/UpscaleScaleSlider';
import { useExpanderToggle } from 'features/settingsAccordions/hooks/useExpanderToggle';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import { UpscaleInitialImage } from './UpscaleInitialImage';
import { UpscaleWarning } from './UpscaleWarning';

const selector = createMemoizedSelector([selectUpscaleSlice], (upscaleSlice) => {
  const { upscaleModel, upscaleInitialImage, scale } = upscaleSlice;

  const badges: string[] = [];

  if (upscaleModel) {
    badges.push(upscaleModel.name);
  }

  if (upscaleInitialImage) {
    const gridSize = upscaleModel ? getGridSize(upscaleModel.base) : getGridSize(null);
    // Output height and width are scaled and rounded down to the nearest multiple of 8
    const outputWidth = roundDownToMultiple(upscaleInitialImage.width * scale, gridSize);
    const outputHeight = roundDownToMultiple(upscaleInitialImage.height * scale, gridSize);

    badges.push(`${outputWidth}×${outputHeight}`);
  }

  return { badges };
});

export const UpscaleSettingsAccordion = memo(() => {
  const { t } = useTranslation();
  const { badges } = useAppSelector(selector);
  const { isOpen: isOpenAccordion, onToggle: onToggleAccordion } = useStandaloneAccordionToggle({
    id: 'upscale-settings',
    defaultIsOpen: true,
  });

  const { isOpen: isOpenExpander, onToggle: onToggleExpander } = useExpanderToggle({
    id: 'upscale-settings-advanced',
    defaultIsOpen: false,
  });

  return (
    <StandaloneAccordion
      label={t('upscaling.upscale')}
      badges={badges}
      isOpen={isOpenAccordion}
      onToggle={onToggleAccordion}
    >
      <Flex pt={4} px={4} w="full" h="full" flexDir="column" data-testid="upscale-settings-accordion">
        <Flex flexDir="column" gap={4}>
          <Flex gap={4}>
            <UpscaleInitialImage />
            <Flex direction="column" w="full" alignItems="center" gap={2} minW={0}>
              <ParamSpandrelModel />
              <UpscaleScaleSlider />
            </Flex>
          </Flex>
          <UpscaleWarning />
        </Flex>
        <Expander label={t('accordions.advanced.options')} isOpen={isOpenExpander} onToggle={onToggleExpander}>
          <Flex gap={4} pb={4} flexDir="column">
            <ParamCreativity />
            <ParamStructure />
          </Flex>
        </Expander>
      </Flex>
    </StandaloneAccordion>
  );
});

UpscaleSettingsAccordion.displayName = 'UpscaleSettingsAccordion';
