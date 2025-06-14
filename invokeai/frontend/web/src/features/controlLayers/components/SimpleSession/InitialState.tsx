/* eslint-disable i18next/no-literal-string */

import { Button, Divider, Flex, Grid, Heading, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { InitialStateAddAStyleReference } from 'features/controlLayers/components/SimpleSession/InitialStateAddAStyleReference';
import { InitialStateEditImageCard } from 'features/controlLayers/components/SimpleSession/InitialStateEditImageCard';
import { InitialStateGenerateFromText } from 'features/controlLayers/components/SimpleSession/InitialStateGenerateFromText';
import { InitialStateUseALayoutImageCard } from 'features/controlLayers/components/SimpleSession/InitialStateUseALayoutImageCard';
import { toast } from 'features/toast/toast';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';

export const InitialState = memo(() => {
  const dispatch = useAppDispatch();
  const newCanvasSession = useCallback(() => {
    dispatch(setActiveTab('canvas'));
    toast({
      title: 'Switched to Canvas',
      description: 'You are in advanced mode yadda yadda.',
      status: 'info',
      position: 'top',
      // isClosable: false,
      duration: 5000,
    });
  }, [dispatch]);

  return (
    <Flex flexDir="column" h="full" w="full" gap={2}>
      <Flex px={2} alignItems="center" minH="24px">
        <Heading size="sm">Get Started</Heading>
      </Flex>
      <Divider />
      <Flex flexDir="column" h="full" justifyContent="center" mx={16}>
        <Heading mb={4}>Choose a starting method.</Heading>
        <Text fontSize="md" fontStyle="italic" mb={6}>
          Drag an image onto a card or click the upload icon.
        </Text>

        <Grid gridTemplateColumns="1fr 1fr" gridTemplateRows="1fr 1fr" gap={4}>
          <InitialStateGenerateFromText />
          <InitialStateAddAStyleReference />
          <InitialStateUseALayoutImageCard />
          <InitialStateEditImageCard />
        </Grid>

        <Text fontSize="md" color="base.300" alignSelf="center" mt={6}>
          or{' '}
          <Button variant="link" onClick={newCanvasSession}>
            start from a blank canvas.
          </Button>
        </Text>
      </Flex>
    </Flex>
  );
});
InitialState.displayName = 'InitialState';
