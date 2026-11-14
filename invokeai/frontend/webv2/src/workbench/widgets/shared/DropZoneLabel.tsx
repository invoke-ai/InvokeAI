import type { ReactNode } from 'react';

import { Text } from '@chakra-ui/react';

/**
 * What a drop zone over Canvas or Preview does, for gallery and desktop drags alike. It sits on its own opaque backing,
 * so it stays legible over the zone's tint and whatever light or busy image shows beneath.
 */
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
