import type { FormLabelProps } from '@invoke-ai/ui-library';
import { Expander, Flex, FormControlGroup, StandaloneAccordion } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasV2Slice } from 'features/controlLayers/store/canvasV2Slice';
import { HrfSettings } from 'features/hrf/components/HrfSettings';
import { selectHrfSlice } from 'features/hrf/store/hrfSlice';
import ParamScaleBeforeProcessing from 'features/parameters/components/Canvas/InfillAndScaling/ParamScaleBeforeProcessing';
import ParamScaledHeight from 'features/parameters/components/Canvas/InfillAndScaling/ParamScaledHeight';
import ParamScaledWidth from 'features/parameters/components/Canvas/InfillAndScaling/ParamScaledWidth';
import ParamImageToImageStrength from 'features/parameters/components/Canvas/ParamImageToImageStrength';
import { ParamSeedNumberInput } from 'features/parameters/components/Seed/ParamSeedNumberInput';
import { ParamSeedRandomize } from 'features/parameters/components/Seed/ParamSeedRandomize';
import { ParamSeedShuffle } from 'features/parameters/components/Seed/ParamSeedShuffle';
import { useExpanderToggle } from 'features/settingsAccordions/hooks/useExpanderToggle';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import { ImageSizeLinear } from './ImageSizeLinear';

const selector = createMemoizedSelector([selectHrfSlice, selectCanvasV2Slice], (hrf, canvasV2) => {
  const { shouldRandomizeSeed, model } = canvasV2.params;
  const { hrfEnabled } = hrf;
  const badges: string[] = [];
  const isSDXL = model?.base === 'sdxl';

  const { aspectRatio, width, height } = canvasV2.document;
  badges.push(`${width}×${height}`);
  badges.push(aspectRatio.id);

  if (aspectRatio.isLocked) {
    badges.push('locked');
  }

  if (!shouldRandomizeSeed) {
    badges.push('Manual Seed');
  }

  if (hrfEnabled && !isSDXL) {
    badges.push('HiRes Fix');
  }
  return { badges, isSDXL };
});

const scalingLabelProps: FormLabelProps = {
  minW: '4.5rem',
};

export const ImageSettingsAccordion = memo(() => {
  const { t } = useTranslation();
  const { badges, isSDXL } = useAppSelector(selector);
  const { isOpen: isOpenAccordion, onToggle: onToggleAccordion } = useStandaloneAccordionToggle({
    id: 'image-settings',
    defaultIsOpen: true,
  });
  const { isOpen: isOpenExpander, onToggle: onToggleExpander } = useExpanderToggle({
    id: 'image-settings-advanced',
    defaultIsOpen: false,
  });

  return (
    <StandaloneAccordion
      label={t('accordions.image.title')}
      badges={badges}
      isOpen={isOpenAccordion}
      onToggle={onToggleAccordion}
    >
      <Flex px={4} pt={4} w="full" h="full" flexDir="column" data-testid="image-settings-accordion">
        <Flex flexDir="column" gap={4}>
          <ImageSizeLinear />
          <ParamImageToImageStrength />
        </Flex>
        <Expander label={t('accordions.advanced.options')} isOpen={isOpenExpander} onToggle={onToggleExpander}>
          <Flex gap={4} pb={4} flexDir="column">
            <Flex gap={4} alignItems="center">
              <ParamSeedNumberInput />
              <ParamSeedShuffle />
              <ParamSeedRandomize />
            </Flex>
            {!isSDXL && <HrfSettings />}
            <ParamScaleBeforeProcessing />
            <FormControlGroup formLabelProps={scalingLabelProps}>
              <ParamScaledWidth />
              <ParamScaledHeight />
            </FormControlGroup>
          </Flex>
        </Expander>
      </Flex>
    </StandaloneAccordion>
  );
});

ImageSettingsAccordion.displayName = 'ImageSettingsAccordion';
