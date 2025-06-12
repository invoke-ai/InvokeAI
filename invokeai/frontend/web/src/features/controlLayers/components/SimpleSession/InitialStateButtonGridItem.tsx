import type { GridItemProps } from '@invoke-ai/ui-library';
import { Button, forwardRef, GridItem } from '@invoke-ai/ui-library';
import { memo } from 'react';

export const InitialStateButtonGridItem = memo(
  forwardRef(({ children, ...rest }: GridItemProps, ref) => {
    return (
      <GridItem
        ref={ref}
        as={Button}
        variant="outline"
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
        {...rest}
      >
        {children}
      </GridItem>
    );
  })
);

InitialStateButtonGridItem.displayName = 'InitialStateButtonGridItem';
