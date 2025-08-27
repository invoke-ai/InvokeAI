import { Alert, Text } from '@invoke-ai/ui-library';
import { memo } from 'react';

export const PixelDimensionsUnsupportedAlert = memo(() => {
  return (
    <Alert status="info" borderRadius="base" flexDir="column" gap={2} overflow="unset">
      <Text fontSize="md">This model does not support user-defined width and height.</Text>
    </Alert>
  );
});

PixelDimensionsUnsupportedAlert.displayName = 'PixelDimensionsUnsupportedAlert';
