import { HStack, Text } from '@chakra-ui/react';

/** Small uppercase section label with an optional count, e.g. "RECENT 13". */
export const SectionHeader = ({ title, count }: { title: string; count?: number }) => (
  <HStack gap="2" px="1">
    <Text color="fg.muted" fontSize="2xs" fontWeight="700" letterSpacing="0.06em" textTransform="uppercase">
      {title}
    </Text>
    {count !== undefined ? (
      <Text color="fg.subtle" fontSize="2xs" fontVariantNumeric="tabular-nums">
        {count}
      </Text>
    ) : null}
  </HStack>
);
