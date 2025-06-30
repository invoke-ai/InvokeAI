import { Button, Flex, Grid, Heading, Text } from '@invoke-ai/ui-library';
import { useAutoLayoutContext } from 'features/ui/layouts/auto-layout-context';
import { WORKSPACE_PANEL_ID } from 'features/ui/layouts/shared';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { InitialStateMainModelPicker } from './InitialStateMainModelPicker';
import { LaunchpadAddStyleReference } from './LaunchpadAddStyleReference';
import { LaunchpadEditImageButton } from './LaunchpadEditImageButton';
import { LaunchpadGenerateFromTextButton } from './LaunchpadGenerateFromTextButton';
import { LaunchpadUseALayoutImageButton } from './LaunchpadUseALayoutImageButton';

export const CanvasLaunchpadPanel = memo(() => {
  const { t } = useTranslation();
  const ctx = useAutoLayoutContext();
  const focusCanvas = useCallback(() => {
    ctx.focusPanel(WORKSPACE_PANEL_ID);
  }, [ctx]);
  return (
    <Flex flexDir="column" h="full" w="full" alignItems="center" gap={2}>
      <Flex flexDir="column" w="full" gap={4} px={14} maxW={768} pt="20vh">
        <Heading mb={4}>{t('ui.launchpad.canvasTitle')}</Heading>
        <Flex flexDir="column" gap={8}>
          <Grid gridTemplateColumns="1fr 1fr" gap={8}>
            <InitialStateMainModelPicker />
            <Flex flexDir="column" gap={2} justifyContent="center">
              <Text>
                Want to learn what prompts work best for each model?{' '}
                <Button as="a" variant="link" href="#" size="sm">
                  Check our our Model Guide.
                </Button>
              </Text>
            </Flex>
          </Grid>
          <LaunchpadGenerateFromTextButton extraAction={focusCanvas} />
          <LaunchpadAddStyleReference extraAction={focusCanvas} />
          <LaunchpadEditImageButton extraAction={focusCanvas} />
          <LaunchpadUseALayoutImageButton extraAction={focusCanvas} />
        </Flex>
      </Flex>
    </Flex>
  );
});
CanvasLaunchpadPanel.displayName = 'CanvasLaunchpadPanel';
