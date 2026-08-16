import { EmptyState as ChakraEmptyState, VStack } from '@chakra-ui/react';
import * as React from 'react';

export interface EmptyStateProps extends ChakraEmptyState.RootProps {
  title: string;
  description?: string | null;
  icon?: React.ReactNode;
  danger?: boolean;
  /**
   * Where the block sits on its own cross axis.
   *
   * `center` (the default) is right inside a dialog, a panel, or anywhere the
   * empty state is the only thing on screen. `start` is right on a page whose
   * heading, description and toolbar are all flush left: a centered block
   * under a left-aligned page has no shared edge with anything above it, and
   * reads as a different screen rather than as the page's own content.
   *
   * `start` also drops the recipe's horizontal padding, so the text lands
   * exactly on the page's text column. Left-aligning but staying inset would
   * be worse than centering — a near miss reads as a mistake, where a clean
   * center reads as a choice.
   */
  align?: 'center' | 'start';
}

export const EmptyState = React.forwardRef<HTMLDivElement, EmptyStateProps>(function EmptyState(props, ref) {
  const { title, description, icon, children, danger, align = 'center', ...rest } = props;
  const fgColor = danger ? 'fg.error' : undefined;
  const isStart = align === 'start';
  return (
    // `px` before the spread, not after: dropping the recipe's inset is this prop's job, but a
    // caller that asked for its own padding still outranks it.
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
