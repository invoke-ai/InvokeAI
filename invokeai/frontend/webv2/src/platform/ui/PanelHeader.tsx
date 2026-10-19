import type { StackProps } from '@chakra-ui/react';

import { HStack } from '@chakra-ui/react';

/** Shared desktop panel chrome; callers supply identity and surface-specific actions. */
export const PanelHeader = (props: StackProps) => (
  <HStack
    justify="space-between"
    borderBottomWidth="1px"
    borderColor="border.subtle"
    h="10"
    minH="10"
    ps="3"
    pe="2"
    {...props}
  />
);
