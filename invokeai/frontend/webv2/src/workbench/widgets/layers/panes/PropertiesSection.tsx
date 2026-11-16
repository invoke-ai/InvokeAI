import type { ReactNode, Ref } from 'react';

import { Flex, Stack, Text, VisuallyHidden } from '@chakra-ui/react';
import { FieldLabel } from '@platform/ui/Field';
import { Tooltip } from '@platform/ui/Tooltip';

/** A headed group of rows in an editor pane: Operation, Tool, Layer. */
export const PropertiesSection = ({
  children,
  disabled = false,
  ref,
  subtitle,
  subtitleDetail,
  title,
  trailing,
}: {
  children: ReactNode;
  /** The surface is busy elsewhere (staging, generation, an operation): the rows stay in place but cannot act. */
  disabled?: boolean;
  ref?: Ref<HTMLDivElement>;
  subtitle?: string;
  /** More about the subtitle: its tooltip, and read with it by assistive tech. */
  subtitleDetail?: string;
  title: string;
  /** Right-aligned beside the title, in the header's one line: it truncates rather than growing the header. */
  trailing?: ReactNode;
}) => {
  const subtitleText = subtitle ? (
    <Text color="fg.muted" flexShrink={1} minW="0" truncate>
      {subtitle}
      {subtitleDetail ? <VisuallyHidden>{subtitleDetail}</VisuallyHidden> : null}
    </Text>
  ) : null;
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
      <Flex align="baseline" gap="2" minW="0">
        <FieldLabel>{title}</FieldLabel>
        {subtitleText && subtitleDetail ? <Tooltip content={subtitleDetail}>{subtitleText}</Tooltip> : subtitleText}
        {trailing}
      </Flex>
      {children}
    </Stack>
  );
};
