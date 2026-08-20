import type { ElementType, ReactNode, Ref } from 'react';

import { Badge, Flex, HStack, Icon, Stack, Text } from '@chakra-ui/react';

export interface KeyStatusBadge {
  label: string;
  palette: string;
}

/**
 * Shared chrome for credential cards: icon tile, title + status badge,
 * description, body.
 *
 * `isHighlighted` is how a card answers "this is the one you asked for" after
 * a jump from elsewhere. One neutral step — the stroke it already has, brought
 * up a stop — rather than a tint or a flash: the grid is a settings surface,
 * and the card still has to be readable while the user types a key into it.
 */
export const KeyCardShell = ({
  children,
  description,
  icon,
  isHighlighted = false,
  ref,
  status,
  title,
}: {
  children: ReactNode;
  description: string;
  icon: ElementType;
  isHighlighted?: boolean;
  ref?: Ref<HTMLDivElement>;
  status: KeyStatusBadge | null;
  title: string;
}) => (
  <Stack
    ref={ref}
    bg="bg.subtle"
    borderColor={isHighlighted ? 'border.emphasized' : 'border.subtle'}
    borderWidth="1px"
    gap="2.5"
    p="3"
    rounded="lg"
    transition="border-color var(--wb-motion-duration-fast) ease"
  >
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
