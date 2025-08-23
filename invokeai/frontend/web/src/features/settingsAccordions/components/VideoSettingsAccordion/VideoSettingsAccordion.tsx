import { Flex, StandaloneAccordion } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  aspectRatioIdChanged,
  aspectRatioLockToggled,
  heightChanged,
  widthChanged,
} from 'features/controlLayers/store/paramsSlice';
import { RUNWAY_ASPECT_RATIOS, VEO3_RESOLUTIONS } from 'features/controlLayers/store/types';
import { Dimensions } from 'features/parameters/components/Dimensions/Dimensions';
import { ParamSeed } from 'features/parameters/components/Seed/ParamSeed';
import { ParamDuration } from 'features/parameters/components/Video/ParamDuration';
import { ParamResolution } from 'features/parameters/components/Video/ParamResolution';
import { selectVideoModel, videoResolutionChanged } from 'features/parameters/store/videoSlice';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { memo, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

import { StartingFrameImage } from './StartingFrameImage';
import { VideoModelPicker } from './VideoModelPicker';

export const VideoSettingsAccordion = memo(() => {
  const { t } = useTranslation();
  const { isOpen: isOpenAccordion, onToggle: onToggleAccordion } = useStandaloneAccordionToggle({
    id: 'video-settings',
    defaultIsOpen: true,
  });
  const videoModel = useAppSelector(selectVideoModel);

  const dispatch = useAppDispatch();

  useEffect(() => {
    // hack to get the default aspect ratio etc for models outside paramsSlice
    if (videoModel?.base === 'runway') {
      dispatch(aspectRatioIdChanged({ id: '16:9' }));
      const { width, height } = RUNWAY_ASPECT_RATIOS['16:9'];
      dispatch(widthChanged({ width, clamp: true }));
      dispatch(heightChanged({ height, clamp: true }));
      dispatch(aspectRatioLockToggled());
    }

    if (videoModel?.base === 'veo3') {
      dispatch(aspectRatioIdChanged({ id: '16:9' }));
      dispatch(videoResolutionChanged('720p'));
      const { width, height } = VEO3_RESOLUTIONS['720p'];
      dispatch(widthChanged({ width, clamp: true }));
      dispatch(heightChanged({ height, clamp: true }));
      dispatch(aspectRatioLockToggled());
    }
  }, [dispatch, videoModel]);

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
