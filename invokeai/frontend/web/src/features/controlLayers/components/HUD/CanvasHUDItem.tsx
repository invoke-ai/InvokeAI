import { GridItem, Text } from '@invoke-ai/ui-library';
import type { Property } from 'csstype';
import { memo } from 'react';

type Props = {
  label: string;
  value: string | number;
  color?: Property.Color;
};

export const CanvasHUDItem = memo(({ label, value, color }: Props) => {
  return (
    <>
      <GridItem>
        <Text textAlign="end">{label}: </Text>
      </GridItem>
      <GridItem fontWeight="semibold">
        <Text textAlign="end" color={color}>
          {value}
        </Text>
      </GridItem>
    </>
  );
});

CanvasHUDItem.displayName = 'CanvasHUDItem';
