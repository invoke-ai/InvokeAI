import { GridItem, Text } from '@invoke-ai/ui-library';
import { memo } from 'react';

export const CanvasHUDItem = memo(({ label, value }: { label: string; value: string | number }) => {
  return (
    <>
      <GridItem>
        <Text textAlign="end">{label}: </Text>
      </GridItem>
      <GridItem fontWeight="semibold">
        <Text textAlign="end">{value}</Text>
      </GridItem>
    </>
  );
});

CanvasHUDItem.displayName = 'CanvasHUDItem';
