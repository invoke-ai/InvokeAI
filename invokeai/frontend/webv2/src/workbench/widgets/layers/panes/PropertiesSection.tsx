import type { ReactNode, Ref } from 'react';

import { Flex, Stack, Text } from '@chakra-ui/react';
import { FieldLabel } from '@platform/ui/Field';

/** A headed group of rows in an editor pane: Operation, Tool, Layer. */
export const PropertiesSection = ({
  children,
  disabled = false,
  headerHidden = false,
  ref,
  subtitle,
  title,
}: {
  children: ReactNode;
  /** The surface is busy elsewhere (staging, generation, an operation): the rows stay in place but cannot act. */
  disabled?: boolean;
  /** The pane title already names this section; `title` still labels the group for assistive technology. */
  headerHidden?: boolean;
  ref?: Ref<HTMLDivElement>;
  subtitle?: string;
  title: string;
}) => {
  return (
    <Stack
      ref={ref}
      aria-label={title}
      borderBottomWidth="1px"
      borderColor="border.subtle"
      gap="2"
      inert={disabled || undefined}
      opacity={disabled ? 0.5 : 1}
      px="3"
      py="2.5"
      role="group"
    >
      {headerHidden ? null : (
        <Flex align="baseline" gap="2" minW="0">
          <FieldLabel>{title}</FieldLabel>
          {subtitle ? (
            <Text color="fg.muted" fontSize="xs" minW="0" truncate>
              {subtitle}
            </Text>
          ) : null}
        </Flex>
      )}
      {children}
    </Stack>
  );
};
