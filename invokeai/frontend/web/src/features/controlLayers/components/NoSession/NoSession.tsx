/* eslint-disable i18next/no-literal-string */

import { Button, Flex, Heading, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { GenerateWithControlImage } from 'features/controlLayers/components/NoSession/GenerateWithControlImage';
import { GenerateWithStartingImage } from 'features/controlLayers/components/NoSession/GenerateWithStartingImage';
import { GenerateWithStartingImageAndInpaintMask } from 'features/controlLayers/components/NoSession/GenerateWithStartingImageAndInpaintMask';
import { canvasSessionStarted } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { memo, useCallback } from 'react';

export const NoSession = memo(() => {
  const dispatch = useAppDispatch();
  const newSesh = useCallback(() => {
    dispatch(canvasSessionStarted({ sessionType: 'advanced' }));
  }, [dispatch]);

  return (
    <Flex flexDir="column" w="full" h="full" alignItems="center" justifyContent="center">
      <Heading>Get Started with Invoke</Heading>
      <Button variant="ghost" onClick={newSesh}>
        Start a new Canvas Session
      </Button>
      <Text>or</Text>
      <Flex flexDir="column" maxW={512}>
        <GenerateWithStartingImage />
        <GenerateWithControlImage />
        <GenerateWithStartingImageAndInpaintMask />
      </Flex>
    </Flex>
  );
});
NoSession.displayName = 'NoSession';
