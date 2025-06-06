import { GridItem } from '@invoke-ai/ui-library';
import { memo, type PropsWithChildren } from 'react';

export const InitialStateCardGridItem = memo((props: PropsWithChildren) => {
  return (
    <GridItem
      display="flex"
      position="relative"
      flexDir="column"
      alignItems="center"
      borderWidth={1}
      borderRadius="base"
      p={2}
      pt={6}
      gap={2}
      w="full"
      h="full"
    >
      {props.children}
    </GridItem>
  );
});

InitialStateCardGridItem.displayName = 'InitialStateCardGridItem';
