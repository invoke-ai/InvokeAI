import type { ReactNode } from 'react';

import { Text } from '@chakra-ui/react';

/** What a drop zone does, on an opaque backing so it stays legible over any image. */
export const DropZoneLabel = ({ children }: { children: ReactNode }) => (
  <Text
    bg="bg.panel"
    color="fg"
    fontSize="lg"
    fontWeight="700"
    maxW="full"
    px="3"
    py="1.5"
    rounded="md"
    shadow="sm"
    textAlign="center"
  >
    {children}
  </Text>
);
