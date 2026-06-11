import { Alert, Box, Flex, Heading, Stack, Text } from '@chakra-ui/react';
import type { ReactNode } from 'react';

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
  <Flex align="center" bg="bg.shell" color="fg.default" justify="center" minH="100dvh" p="6">
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
      <Stack bg="bg.surface" borderColor="border.subtle" borderWidth="1px" gap="4" p="6" rounded="xl" shadow="lg">
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

const InvokeMark = () => (
  <Box color="accent.invoke">
    <svg aria-hidden="true" fill="none" height="36" viewBox="0 0 44 44" width="36">
      <path
        d="M29.1951 10.6667H42V2H2V10.6667H14.8049L29.1951 33.3333H42V42H2V33.3333H14.8049"
        stroke="currentColor"
        strokeWidth="2.8"
      />
    </svg>
  </Box>
);
