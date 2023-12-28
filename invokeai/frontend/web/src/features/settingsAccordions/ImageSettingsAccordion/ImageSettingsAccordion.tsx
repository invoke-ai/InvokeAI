import { Flex } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { InvControlGroup } from 'common/components/InvControl/InvControlGroup';
import type { InvLabelProps } from 'common/components/InvControl/types';
import { InvExpander } from 'common/components/InvExpander/InvExpander';
import { InvSingleAccordion } from 'common/components/InvSingleAccordion/InvSingleAccordion';
import { HrfSettings } from 'features/hrf/components/HrfSettings';
import ParamBoundingBoxHeight from 'features/parameters/components/Canvas/BoundingBox/ParamBoundingBoxHeight';
import ParamBoundingBoxWidth from 'features/parameters/components/Canvas/BoundingBox/ParamBoundingBoxWidth';
import { ParamHeight } from 'features/parameters/components/Core/ParamHeight';
import { ParamWidth } from 'features/parameters/components/Core/ParamWidth';
import { AspectRatioPreviewWrapper } from 'features/parameters/components/ImageSize/AspectRatioPreviewWrapper';
import { AspectRatioSelect } from 'features/parameters/components/ImageSize/AspectRatioSelect';
import ImageToImageFit from 'features/parameters/components/ImageToImage/ImageToImageFit';
import ImageToImageStrength from 'features/parameters/components/ImageToImage/ImageToImageStrength';
import { ParamSeedNumberInput } from 'features/parameters/components/Seed/ParamSeedNumberInput';
import { ParamSeedRandomize } from 'features/parameters/components/Seed/ParamSeedRandomize';
import { ParamSeedShuffle } from 'features/parameters/components/Seed/ParamSeedShuffle';
import type { InvokeTabName } from 'features/ui/store/tabMap';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(
  [stateSelector, activeTabNameSelector],
  ({ generation, hrf }, activeTabName) => {
    const { aspectRatio, width, height, shouldRandomizeSeed } = generation;
    const { hrfEnabled } = hrf;
    const badges = [`${width}×${height}`, aspectRatio.id];
    if (!shouldRandomizeSeed) {
      badges.push('Manual Seed');
    }
    if (hrfEnabled) {
      badges.push('HiRes Fix');
    }
    return { badges, activeTabName };
  }
);

const labelProps: InvLabelProps = {
  w: 12,
};

export const ImageSettingsAccordion = () => {
  const { t } = useTranslation();
  const { badges, activeTabName } = useAppSelector(selector);

  return (
    <InvSingleAccordion
      label={t('accordions.image.title')}
      defaultIsOpen={true}
      badges={badges}
    >
      <Flex px={4} pt={4} w="full" h="full" flexDir="column">
        <Flex gap={4} alignItems="center">
          <Flex gap={4} flexDirection="column" width="full">
            <InvControlGroup labelProps={labelProps}>
              <AspectRatioSelect />
              <WidthHeight activeTabName={activeTabName} />
            </InvControlGroup>
          </Flex>
          <Flex w="98px" h="98px" flexShrink={0} flexGrow={0}>
            <AspectRatioPreviewWrapper />
          </Flex>
        </Flex>
        <InvExpander>
          <Flex gap={4} pb={4} flexDir="column">
            <Flex gap={4}>
              <ParamSeedNumberInput />
              <ParamSeedShuffle />
              <ParamSeedRandomize />
            </Flex>
            {activeTabName === 'txt2img' && <HrfSettings />}
            {activeTabName === 'img2img' && <ImageToImageFit />}
            {activeTabName === 'img2img' && <ImageToImageStrength />}
          </Flex>
        </InvExpander>
      </Flex>
    </InvSingleAccordion>
  );
};

const WidthHeight = (props: { activeTabName: InvokeTabName }) => {
  if (props.activeTabName === 'unifiedCanvas') {
    return (
      <>
        <ParamBoundingBoxWidth />
        <ParamBoundingBoxHeight />
      </>
    );
  }

  return (
    <>
      <ParamWidth />
      <ParamHeight />
    </>
  );
};
