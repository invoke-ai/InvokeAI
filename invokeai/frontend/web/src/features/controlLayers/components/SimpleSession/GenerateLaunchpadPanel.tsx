import { Alert, Button, Flex, Grid, Text } from '@invoke-ai/ui-library';
import { InitialStateMainModelPicker } from 'features/controlLayers/components/SimpleSession/InitialStateMainModelPicker';
import { LaunchpadAddStyleReference } from 'features/controlLayers/components/SimpleSession/LaunchpadAddStyleReference';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { memo, useCallback } from 'react';

import { LaunchpadContainer } from './LaunchpadContainer';
import { LaunchpadGenerateFromTextButton } from './LaunchpadGenerateFromTextButton';

export const GenerateLaunchpadPanel = memo(() => {
  const newCanvasSession = useCallback(() => {
    navigationApi.switchToTab('canvas');
  }, []);

  return (
    <LaunchpadContainer heading="Generate images from text prompts.">
      <Grid gridTemplateColumns="1fr 1fr" gap={8}>
        <InitialStateMainModelPicker />
        <Flex flexDir="column" gap={2} justifyContent="center">
          <Text>
            Want to learn what prompts work best for each model?{' '}
            <Button
              as="a"
              variant="link"
              href="https://support.invoke.ai/support/solutions/articles/151000216086-model-guide"
              target="_blank"
              rel="noopener noreferrer"
              size="sm"
            >
              Check out our Model Guide.
            </Button>
          </Text>
        </Flex>
      </Grid>
      <LaunchpadGenerateFromTextButton />
      <LaunchpadAddStyleReference />
      <Alert status="info" borderRadius="base" flexDir="column" gap={2} overflow="unset">
        <Text fontSize="md" fontWeight="semibold">
          Looking to get more control, edit, and iterate on your images?
        </Text>
        <Button variant="link" onClick={newCanvasSession}>
          Navigate to Canvas for more capabilities.
        </Button>
      </Alert>
    </LaunchpadContainer>
  );
});
GenerateLaunchpadPanel.displayName = 'GenerateLaunchpad';
