import type { ElementType, ReactNode } from 'react';

import { Badge, Flex, HStack, Icon, Stack, Text } from '@chakra-ui/react';

export interface KeyStatusBadge {
  label: string;
  palette: string;
}

/** Shared chrome for credential cards: icon tile, title + status badge, description, body. */
export const KeyCardShell = ({
  children,
  description,
  icon,
  status,
  title,
}: {
  children: ReactNode;
  description: string;
  icon: ElementType;
  status: KeyStatusBadge | null;
  title: string;
}) => (
  <Stack bg="bg.subtle" borderColor="border.subtle" borderWidth="1px" gap="2.5" p="3" rounded="lg">
    <HStack align="start" gap="2.5">
      <Flex
        align="center"
        bg="bg.emphasized"
        borderColor="border.subtle"
        borderWidth="1px"
        boxSize="8"
        color="fg.muted"
        flexShrink={0}
        justify="center"
        rounded="md"
      >
        <Icon as={icon} boxSize="4" />
      </Flex>
      <Stack flex="1" gap="0" minW="0">
        <HStack gap="1.5">
          <Text fontSize="xs" fontWeight="700" truncate>
            {title}
          </Text>
          {status ? (
            <Badge colorPalette={status.palette} flexShrink={0} fontSize="2xs" size="sm" variant="surface">
              {status.label}
            </Badge>
          ) : null}
        </HStack>
        <Text color="fg.subtle" fontSize="2xs" lineClamp={2}>
          {description}
        </Text>
      </Stack>
    </HStack>
    {children}
  </Stack>
);
