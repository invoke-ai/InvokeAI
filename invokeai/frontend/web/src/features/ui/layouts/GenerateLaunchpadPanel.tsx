import { Alert, Button, Flex, Grid, Text } from '@invoke-ai/ui-library';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { InitialStateMainModelPicker } from './InitialStateMainModelPicker';
import { LaunchpadAddStyleReference } from './LaunchpadAddStyleReference';
import { LaunchpadContainer } from './LaunchpadContainer';
import { LaunchpadGenerateFromTextButton } from './LaunchpadGenerateFromTextButton';

export const GenerateLaunchpadPanel = memo(() => {
  const { t } = useTranslation();
  const newCanvasSession = useCallback(() => {
    navigationApi.switchToTab('canvas');
  }, []);

  return (
    <LaunchpadContainer heading={t('ui.launchpad.generateTitle')}>
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
      <LaunchpadGenerateFromTextButton />
      <LaunchpadAddStyleReference />
      <Alert status="info" borderRadius="base" flexDir="column" gap={2} overflow="unset">
        <Text fontSize="md" fontWeight="semibold">
          {t('ui.launchpad.generate.canvasCalloutTitle')}
        </Text>
        <Button variant="link" onClick={newCanvasSession}>
          {t('ui.launchpad.generate.canvasCalloutLink')}
        </Button>
      </Alert>
    </LaunchpadContainer>
  );
});
GenerateLaunchpadPanel.displayName = 'GenerateLaunchpad';
