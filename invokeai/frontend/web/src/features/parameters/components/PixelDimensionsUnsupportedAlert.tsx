import { Alert, Text } from '@invoke-ai/ui-library';
import { memo } from 'react';

export const PixelDimensionsUnsupportedAlert = memo(() => {
  return (
    <Alert status="info" borderRadius="base" flexDir="column" gap={2} overflow="unset">
      <Text fontSize="sm" color="base.100">
        Select an aspect ratio to control the size of the resulting image from this model.
      </Text>
    </Alert>
  );
});

PixelDimensionsUnsupportedAlert.displayName = 'PixelDimensionsUnsupportedAlert';
