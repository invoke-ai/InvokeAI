import { EmptyState as ChakraEmptyState, VStack } from '@chakra-ui/react';
import * as React from 'react';

export interface EmptyStateProps extends ChakraEmptyState.RootProps {
  title: string;
  description?: string | null;
  icon?: React.ReactNode;
  danger?: boolean;
  /** center suits isolated empty states; start aligns with page content and removes horizontal recipe padding. */
  align?: 'center' | 'start';
}

export const EmptyState = React.forwardRef<HTMLDivElement, EmptyStateProps>(function EmptyState(props, ref) {
  const { title, description, icon, children, danger, align = 'center', ...rest } = props;
  const fgColor = danger ? 'fg.error' : undefined;
  const isStart = align === 'start';
  return (
    // Apply default px before caller props so explicit padding wins.
    <ChakraEmptyState.Root ref={ref} px={isStart ? '0' : undefined} {...rest} size={props.size ?? 'sm'}>
      <ChakraEmptyState.Content alignItems={isStart ? 'flex-start' : undefined}>
        {icon && <ChakraEmptyState.Indicator color={fgColor}>{icon}</ChakraEmptyState.Indicator>}
        {description ? (
          <VStack align={isStart ? 'start' : 'center'} textAlign={isStart ? 'start' : 'center'} color={fgColor}>
            <ChakraEmptyState.Title>{title}</ChakraEmptyState.Title>
            <ChakraEmptyState.Description>{description}</ChakraEmptyState.Description>
          </VStack>
        ) : (
          // A title long enough to wrap aligns with the rest of the block, not against it.
          <ChakraEmptyState.Title color={fgColor} textAlign={isStart ? 'start' : undefined}>
            {title}
          </ChakraEmptyState.Title>
        )}
        {children}
      </ChakraEmptyState.Content>
    </ChakraEmptyState.Root>
  );
});
