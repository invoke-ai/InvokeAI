import { Button, Flex, Grid, Text } from '@invoke-ai/ui-library';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { WORKSPACE_PANEL_ID } from 'features/ui/layouts/shared';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { InitialStateMainModelPicker } from './InitialStateMainModelPicker';
import { LaunchpadAddStyleReference } from './LaunchpadAddStyleReference';
import { LaunchpadContainer } from './LaunchpadContainer';
import { LaunchpadEditImageButton } from './LaunchpadEditImageButton';
import { LaunchpadGenerateFromTextButton } from './LaunchpadGenerateFromTextButton';
import { LaunchpadUseALayoutImageButton } from './LaunchpadUseALayoutImageButton';

export const CanvasLaunchpadPanel = memo(() => {
  const { t } = useTranslation();
  const focusCanvas = useCallback(() => {
    navigationApi.focusPanel('canvas', WORKSPACE_PANEL_ID);
  }, []);
  return (
    <LaunchpadContainer heading={t('ui.launchpad.canvasTitle')}>
      <Grid gridTemplateColumns="1fr 1fr" gap={8}>
        <InitialStateMainModelPicker />
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
      <LaunchpadGenerateFromTextButton extraAction={focusCanvas} />
      <LaunchpadAddStyleReference extraAction={focusCanvas} />
      <LaunchpadEditImageButton extraAction={focusCanvas} />
      <LaunchpadUseALayoutImageButton extraAction={focusCanvas} />
    </LaunchpadContainer>
  );
});
CanvasLaunchpadPanel.displayName = 'CanvasLaunchpadPanel';
