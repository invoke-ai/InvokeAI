import { Button, Flex, Grid, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $videoUpsellComponent } from 'app/store/nanostores/videoUpsellComponent';
import { useAppSelector } from 'app/store/storeHooks';
import { VideoModelPicker } from 'features/settingsAccordions/components/VideoSettingsAccordion/VideoModelPicker';
import { selectAllowVideo } from 'features/system/store/configSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import { LaunchpadContainer } from './LaunchpadContainer';
import { LaunchpadGenerateFromTextButton } from './LaunchpadGenerateFromTextButton';
import { LaunchpadStartingFrameButton } from './LaunchpadStartingFrameButton';

export const VideoLaunchpadPanel = memo(() => {
  const { t } = useTranslation();
  const isVideoEnabled = useAppSelector(selectAllowVideo);
  const videoUpsellComponent = useStore($videoUpsellComponent);

  if (!isVideoEnabled) {
    return (
      <LaunchpadContainer heading="">
          {videoUpsellComponent}
       
      </LaunchpadContainer>
    );
  }

  return (
    <LaunchpadContainer heading={t('ui.launchpad.videoTitle')}>
      <Grid gridTemplateColumns="1fr 1fr" gap={8}>
        <VideoModelPicker />
        <Flex flexDir="column" gap={2} justifyContent="center">
          <Text>
            {t('ui.launchpad.modelGuideText')}{' '}
            <Button
              as="a"
              variant="link"
              href="https://support.invoke.ai/support/solutions/articles/151000216086-model-guide"
              target="_blank"
              rel="noopener noreferrer"
              size="sm"
            >
              {t('ui.launchpad.modelGuideLink')}
            </Button>
          </Text>
        </Flex>
      </Grid>
      <LaunchpadGenerateFromTextButton />
      <LaunchpadStartingFrameButton />
    </LaunchpadContainer>
  );
});
VideoLaunchpadPanel.displayName = 'VideoLaunchpadPanel';
