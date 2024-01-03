import { Flex } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { InvControlGroup } from 'common/components/InvControl/InvControlGroup';
import type { InvLabelProps } from 'common/components/InvControl/types';
import { InvExpander } from 'common/components/InvExpander/InvExpander';
import { InvSingleAccordion } from 'common/components/InvSingleAccordion/InvSingleAccordion';
import { HrfSettings } from 'features/hrf/components/HrfSettings';
import ParamScaleBeforeProcessing from 'features/parameters/components/Canvas/InfillAndScaling/ParamScaleBeforeProcessing';
import ParamScaledHeight from 'features/parameters/components/Canvas/InfillAndScaling/ParamScaledHeight';
import ParamScaledWidth from 'features/parameters/components/Canvas/InfillAndScaling/ParamScaledWidth';
import ImageToImageFit from 'features/parameters/components/ImageToImage/ImageToImageFit';
import ImageToImageStrength from 'features/parameters/components/ImageToImage/ImageToImageStrength';
import { ParamSeedNumberInput } from 'features/parameters/components/Seed/ParamSeedNumberInput';
import { ParamSeedRandomize } from 'features/parameters/components/Seed/ParamSeedRandomize';
import { ParamSeedShuffle } from 'features/parameters/components/Seed/ParamSeedShuffle';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import { ImageSizeCanvas } from './ImageSizeCanvas';
import { ImageSizeLinear } from './ImageSizeLinear';

const selector = createMemoizedSelector(
  [stateSelector, activeTabNameSelector],
  ({ generation, canvas, hrf }, activeTabName) => {
    const { shouldRandomizeSeed } = generation;
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
    return { badges, activeTabName };
  }
);

const scalingLabelProps: InvLabelProps = {
  minW: '4.5rem',
};

export const ImageSettingsAccordion = memo(() => {
  const { t } = useTranslation();
  const { badges, activeTabName } = useAppSelector(selector);

  return (
    <InvSingleAccordion
      label={t('accordions.image.title')}
      defaultIsOpen={true}
      badges={badges}
    >
      <Flex px={4} pt={4} w="full" h="full" flexDir="column">
        {activeTabName === 'unifiedCanvas' ? (
          <ImageSizeCanvas />
        ) : (
          <ImageSizeLinear />
        )}
        <InvExpander>
          <Flex
            flexDirection="column"
            gap={4}
            p={4}
            mb={4}
            borderRadius={4}
            background="base.750"
          >
            <Flex gap={4}>
              <ParamSeedNumberInput />
              <ParamSeedShuffle />
              <ParamSeedRandomize />
            </Flex>
            {(activeTabName === 'img2img' ||
              activeTabName === 'unifiedCanvas') && <ImageToImageStrength />}
            {activeTabName === 'img2img' && <ImageToImageFit />}
            {activeTabName === 'txt2img' && <HrfSettings />}
            {activeTabName === 'unifiedCanvas' && (
              <>
                <ParamScaleBeforeProcessing />
                <InvControlGroup labelProps={scalingLabelProps}>
                  <ParamScaledWidth />
                  <ParamScaledHeight />
                </InvControlGroup>
              </>
            )}
          </Flex>
        </InvExpander>
      </Flex>
    </InvSingleAccordion>
  );
});

ImageSettingsAccordion.displayName = 'ImageSettingsAccordion';
