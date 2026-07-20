import { EmptyState as ChakraEmptyState, VStack } from '@chakra-ui/react';
import * as React from 'react';

export interface EmptyStateProps extends ChakraEmptyState.RootProps {
  title: string;
  description?: string | null;
  icon?: React.ReactNode;
  danger?: boolean;
}

export const EmptyState = React.forwardRef<HTMLDivElement, EmptyStateProps>(function EmptyState(props, ref) {
  const { title, description, icon, children, danger, ...rest } = props;
  const fgColor = danger ? 'fg.error' : undefined;
  return (
    <ChakraEmptyState.Root ref={ref} {...rest} size={props.size ?? 'sm'}>
      <ChakraEmptyState.Content>
        {icon && <ChakraEmptyState.Indicator color={fgColor}>{icon}</ChakraEmptyState.Indicator>}
        {description ? (
          <VStack textAlign="center" color={fgColor}>
            <ChakraEmptyState.Title>{title}</ChakraEmptyState.Title>
            <ChakraEmptyState.Description>{description}</ChakraEmptyState.Description>
          </VStack>
        ) : (
          <ChakraEmptyState.Title color={fgColor}>{title}</ChakraEmptyState.Title>
        )}
        {children}
      </ChakraEmptyState.Content>
    </ChakraEmptyState.Root>
  );
});
