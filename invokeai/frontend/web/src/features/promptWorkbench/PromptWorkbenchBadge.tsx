import { Badge } from '@invoke-ai/ui-library';
import type { PropsWithChildren } from 'react';
import { forwardRef, memo } from 'react';

export type PromptWorkbenchBadgeTone = 'neutral' | 'warning' | 'error';

type PromptWorkbenchBadgeProps = PropsWithChildren<{
  tone?: PromptWorkbenchBadgeTone;
}>;

export const PromptWorkbenchBadge = memo(forwardRef<HTMLSpanElement, PromptWorkbenchBadgeProps>(({ children, tone = 'neutral' }, ref) => {
  const colors = getBadgeColors(tone);

  return (
    <Badge
      ref={ref}
      size="sm"
      bg="transparent"
      borderWidth={1}
      borderColor={colors.borderColor}
      color={colors.color}
      borderRadius="base"
      textTransform="none"
      fontSize="xs"
      fontWeight="semibold"
      lineHeight="short"
      minH={5}
      px={1.5}
      flexShrink={0}
    >
      {children}
    </Badge>
  );
}));

PromptWorkbenchBadge.displayName = 'PromptWorkbenchBadge';

const getBadgeColors = (tone: PromptWorkbenchBadgeTone): { color: string; borderColor: string } => {
  switch (tone) {
    case 'warning':
      return { color: 'warning.300', borderColor: 'warning.700' };
    case 'error':
      return { color: 'error.300', borderColor: 'error.700' };
    case 'neutral':
      return { color: 'base.200', borderColor: 'base.600' };
  }
};
