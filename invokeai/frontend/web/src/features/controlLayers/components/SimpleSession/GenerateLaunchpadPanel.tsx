import { Alert, Button, Flex, Grid, Heading, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { InitialStateMainModelPicker } from 'features/controlLayers/components/SimpleSession/InitialStateMainModelPicker';
import { LaunchpadAddStyleReference } from 'features/controlLayers/components/SimpleSession/LaunchpadAddStyleReference';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { LaunchpadGenerateFromTextButton } from './LaunchpadGenerateFromTextButton';

export const GenerateLaunchpadPanel = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const newCanvasSession = useCallback(() => {
    dispatch(setActiveTab('canvas'));
  }, [dispatch]);

  return (
    <Flex flexDir="column" h="full" w="full" alignItems="center" gap={2}>
      <Flex flexDir="column" w="full" gap={4} px={14} maxW={768} pt="20vh">
        <Heading mb={4}>{t('ui.launchpad.generateTitle')}</Heading>
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
          <Alert status="info" borderRadius="base" flexDir="column" gap={2} overflow="unset">
            <Text fontSize="md" fontWeight="semibold">
              Looking to get more control, edit, and iterate on your images?
            </Text>
            <Button variant="link" onClick={newCanvasSession}>
              Navigate to Canvas for more capabilities.
            </Button>
          </Alert>
        </Flex>
      </Flex>
    </Flex>
  );
});
GenerateLaunchpadPanel.displayName = 'GenerateLaunchpad';
