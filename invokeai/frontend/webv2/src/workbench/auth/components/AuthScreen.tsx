import type { ReactNode } from 'react';

import { Alert, Flex, Heading, Stack, Text } from '@chakra-ui/react';
import { InvokeMark } from '@workbench/components/InvokeMark';

/**
 * Centered scaffold shared by the login and first-run setup screens. These
 * render outside the workbench shell (and its providers), so they rely only on
 * semantic tokens.
 */
export const AuthScreen = ({
  children,
  footer,
  subtitle,
  title,
}: {
  children: ReactNode;
  footer?: ReactNode;
  subtitle: string;
  title: string;
}) => (
  <Flex align="center" bg="bg" color="fg" justify="center" minH="100dvh" p="6">
    <Stack gap="6" maxW="sm" w="full">
      <Stack align="center" gap="4">
        <InvokeMark />
        <Stack align="center" gap="1">
          <Heading fontSize="lg" fontWeight="700">
            {title}
          </Heading>
          <Text color="fg.muted" fontSize="sm" textAlign="center">
            {subtitle}
          </Text>
        </Stack>
      </Stack>
      <Stack bg="bg.subtle" borderColor="border.subtle" borderWidth="1px" gap="4" p="6" rounded="xl" shadow="lg">
        {children}
      </Stack>
      {footer ? (
        <Text color="fg.subtle" fontSize="xs" textAlign="center">
          {footer}
        </Text>
      ) : null}
    </Stack>
  </Flex>
);

/** Inline alert used for form-level failures on the auth screens. */
export const AuthFormAlert = ({ message, tone }: { message: string; tone: 'error' | 'warning' }) => (
  <Alert.Root borderRadius="md" size="sm" status={tone}>
    <Alert.Indicator />
    <Alert.Title fontSize="xs">{message}</Alert.Title>
  </Alert.Root>
);
