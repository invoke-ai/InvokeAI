import { Button, Flex, Grid, Heading, Text } from '@invoke-ai/ui-library';
import { memo } from 'react';

import { InitialStateMainModelPicker } from './InitialStateMainModelPicker';
import { LaunchpadAddStyleReference } from './LaunchpadAddStyleReference';
import { LaunchpadEditImageButton } from './LaunchpadEditImageButton';
import { LaunchpadGenerateFromTextButton } from './LaunchpadGenerateFromTextButton';
import { LaunchpadUseALayoutImageButton } from './LaunchpadUseALayoutImageButton';

export const CanvasLaunchpadPanel = memo(() => {
  return (
    <Flex flexDir="column" h="full" w="full" alignItems="center" gap={2}>
      <Flex flexDir="column" w="full" h="full" gap={4} px={14} maxW={768} pt="20%">
        <Heading mb={4}>Edit and refine on Canvas.</Heading>
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
          <LaunchpadGenerateFromTextButton />
          <LaunchpadAddStyleReference />
          <LaunchpadEditImageButton />
          <LaunchpadUseALayoutImageButton />
        </Flex>
      </Flex>
    </Flex>
  );
});
CanvasLaunchpadPanel.displayName = 'CanvasLaunchpadPanel';
