import type { FormLabelProps } from '@invoke-ai/ui-library';
import { Expander, Flex, FormControlGroup, StandaloneAccordion } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasSlice } from 'features/canvas/store/canvasSlice';
import { HrfSettings } from 'features/hrf/components/HrfSettings';
import { selectHrfSlice } from 'features/hrf/store/hrfSlice';
import ParamScaleBeforeProcessing from 'features/parameters/components/Canvas/InfillAndScaling/ParamScaleBeforeProcessing';
import ParamScaledHeight from 'features/parameters/components/Canvas/InfillAndScaling/ParamScaledHeight';
import ParamScaledWidth from 'features/parameters/components/Canvas/InfillAndScaling/ParamScaledWidth';
import ImageToImageFit from 'features/parameters/components/ImageToImage/ImageToImageFit';
import ImageToImageStrength from 'features/parameters/components/ImageToImage/ImageToImageStrength';
import { ParamSeedNumberInput } from 'features/parameters/components/Seed/ParamSeedNumberInput';
import { ParamSeedRandomize } from 'features/parameters/components/Seed/ParamSeedRandomize';
import { ParamSeedShuffle } from 'features/parameters/components/Seed/ParamSeedShuffle';
import { selectGenerationSlice } from 'features/parameters/store/generationSlice';
import { useExpanderToggle } from 'features/settingsAccordions/hooks/useExpanderToggle';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import { ImageSizeCanvas } from './ImageSizeCanvas';
import { ImageSizeLinear } from './ImageSizeLinear';

const selector = createMemoizedSelector(
  [selectGenerationSlice, selectCanvasSlice, selectHrfSlice, activeTabNameSelector],
  (generation, canvas, hrf, activeTabName) => {
    const { shouldRandomizeSeed, model } = generation;
    const { hrfEnabled } = hrf;
    const badges: string[] = [];

    if (activeTabName === 'unifiedCanvas') {
      const {
        aspectRatio,
        boundingBoxDimensions: { width, height },
      } = canvas;
      badges.push(`${width}×${height}`);
      badges.push(aspectRatio.id);
      if (aspectRatio.isLocked) {
        badges.push('locked');
      }
    } else {
      const { aspectRatio, width, height } = generation;
      badges.push(`${width}×${height}`);
      badges.push(aspectRatio.id);
      if (aspectRatio.isLocked) {
        badges.push('locked');
      }
    }

    if (!shouldRandomizeSeed) {
      badges.push('Manual Seed');
    }

    if (hrfEnabled) {
      badges.push('HiRes Fix');
    }
    return { badges, activeTabName, isSDXL: model?.base === 'sdxl' };
  }
);

const scalingLabelProps: FormLabelProps = {
  minW: '4.5rem',
};

export const ImageSettingsAccordion = memo(() => {
  const { t } = useTranslation();
  const { badges, activeTabName, isSDXL } = useAppSelector(selector);
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
      <Flex px={4} pt={4} w="full" h="full" flexDir="column">
        {activeTabName === 'unifiedCanvas' ? <ImageSizeCanvas /> : <ImageSizeLinear />}
        <Expander label={t('accordions.advanced.options')} isOpen={isOpenExpander} onToggle={onToggleExpander}>
          <Flex gap={4} pb={4} flexDir="column">
            <Flex gap={4} alignItems="center">
              <ParamSeedNumberInput />
              <ParamSeedShuffle />
              <ParamSeedRandomize />
            </Flex>
            {(activeTabName === 'img2img' || activeTabName === 'unifiedCanvas') && <ImageToImageStrength />}
            {activeTabName === 'img2img' && <ImageToImageFit />}
            {activeTabName === 'txt2img' && !isSDXL && <HrfSettings />}
            {activeTabName === 'unifiedCanvas' && (
              <>
                <ParamScaleBeforeProcessing />
                <FormControlGroup formLabelProps={scalingLabelProps}>
                  <ParamScaledWidth />
                  <ParamScaledHeight />
                </FormControlGroup>
              </>
            )}
          </Flex>
        </Expander>
      </Flex>
    </StandaloneAccordion>
  );
});

ImageSettingsAccordion.displayName = 'ImageSettingsAccordion';
