import { Alert, Button, Flex, Grid, Heading, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { InitialStateAddAStyleReference } from 'features/controlLayers/components/SimpleSession/InitialStateAddAStyleReference';
import { InitialStateGenerateFromText } from 'features/controlLayers/components/SimpleSession/InitialStateGenerateFromText';
import { InitialStateMainModelPicker } from 'features/controlLayers/components/SimpleSession/InitialStateMainModelPicker';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';

export const InitialState = memo(() => {
  const dispatch = useAppDispatch();
  const newCanvasSession = useCallback(() => {
    dispatch(setActiveTab('canvas'));
  }, [dispatch]);

  return (
    <Flex flexDir="column" h="full" w="full" alignItems="center" justifyContent="center" gap={2}>
      <Flex flexDir="column" w="full" h="full" justifyContent="center" gap={4} px={12} maxW={768}>
        <Heading mb={4}>Get started with Invoke.</Heading>
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
          <InitialStateGenerateFromText />
          <InitialStateAddAStyleReference />
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
InitialState.displayName = 'InitialState';
