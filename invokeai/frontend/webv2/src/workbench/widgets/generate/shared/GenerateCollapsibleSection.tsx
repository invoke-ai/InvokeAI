import { Box, Collapsible, Flex, Text } from '@chakra-ui/react';
import { ChevronRightIcon } from 'lucide-react';

type Props = {
  label: string;
  isOpen?: boolean;
  defaultOpen?: boolean;
  badges?: React.ReactNode;
  children: React.ReactNode;
};

export const GenerateCollapsibleSection = ({ label, defaultOpen, isOpen, children, badges }: Props) => {
  return (
    <Collapsible.Root defaultOpen={defaultOpen} open={isOpen} bg="bg.muted/50" rounded="md" overflow="hidden">
      <Collapsible.Trigger display="flex" gap={2} w="full" p={2} alignItems="center">
        <Collapsible.Indicator _open={{ transform: 'rotate(90deg)' }} transition="transform 0.2s">
          <ChevronRightIcon size="14" />
        </Collapsible.Indicator>
        <Text
          as="span"
          fontSize="2xs"
          truncate
          letterSpacing="widest"
          fontWeight="bold"
          textTransform="uppercase"
          color="fg.muted"
          lineHeight="1"
        >
          {label}
        </Text>

        <Flex gap={1} ml="auto" fontFamily="mono">
          {badges}
        </Flex>
      </Collapsible.Trigger>
      <Collapsible.Content bg="bg.muted">
        <Box borderTopWidth={1} borderColor="bg.subtle">
          {children}
        </Box>
      </Collapsible.Content>
    </Collapsible.Root>
  );
};
