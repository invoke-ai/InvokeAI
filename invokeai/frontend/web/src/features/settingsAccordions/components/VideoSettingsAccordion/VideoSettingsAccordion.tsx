import { Flex, StandaloneAccordion } from '@invoke-ai/ui-library';
import { Dimensions } from 'features/parameters/components/Dimensions/Dimensions';
import { ParamSeed } from 'features/parameters/components/Seed/ParamSeed';
import { ParamDuration } from 'features/parameters/components/Video/ParamDuration';
import { ParamResolution } from 'features/parameters/components/Video/ParamResolution';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import { StartingFrameImage } from './StartingFrameImage';
import { VideoModelPicker } from './VideoModelPicker';

export const VideoSettingsAccordion = memo(() => {
  const { t } = useTranslation();
  const { isOpen: isOpenAccordion, onToggle: onToggleAccordion } = useStandaloneAccordionToggle({
    id: 'video-settings',
    defaultIsOpen: true,
  });
  return (
    <StandaloneAccordion
      label={t('parameters.video')}
      badges={[]}
      isOpen={isOpenAccordion}
      onToggle={onToggleAccordion}
    >
      <Flex p={4} w="full" h="full" flexDir="column" data-testid="upscale-settings-accordion">
        <Flex gap={4} flexDirection="column" width="full">
          <Flex gap={4}>
            <StartingFrameImage />
            <Flex gap={4} flexDirection="column" width="full">
              <VideoModelPicker />
              <ParamDuration />
              <ParamResolution />
            </Flex>
          </Flex>
          <Dimensions />
          <ParamSeed />
        </Flex>
      </Flex>
    </StandaloneAccordion>
  );
});

VideoSettingsAccordion.displayName = 'VideoSettingsAccordion';
