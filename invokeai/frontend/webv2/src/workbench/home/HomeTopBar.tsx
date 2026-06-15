import { Flex, HStack, Text } from '@chakra-ui/react';

import { AccountMenu } from '@workbench/auth/components/AccountMenu';
import { InvokeMark } from '@workbench/auth/components/AuthScreen';

/**
 * Home's slim header: brand on the left, shared account/settings controls on the right.
 */
export const HomeTopBar = () => (
  <Flex
    align="center"
    borderBottomWidth="1px"
    borderColor="border.subtle"
    flexShrink={0}
    justify="space-between"
    px={{ base: '5', md: '8' }}
    py="3"
  >
    <HStack gap="2.5">
      <InvokeMark size={22} />
      <Text fontSize="sm" fontWeight="700">
        Invoke
      </Text>
    </HStack>
    <AccountMenu />
  </Flex>
);
