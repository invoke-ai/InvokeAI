import { HStack, Icon, Text, VisuallyHidden, type StackProps } from '@chakra-ui/react';
import { RetryButton } from '@platform/ui/LoadError';
import { TriangleAlertIcon } from 'lucide-react';

import type { GalleryReadState } from './galleryStateView';

const OPERABLE = ['button', '[href]', 'input', 'select', 'textarea', '[tabindex]']
  .map((selector) => `${selector}:not([disabled]):not([tabindex="-1"]):not([aria-hidden="true"])`)
  .join(', ');

/**
 * Focuses the first (or last) element matching `selector` (any operable element by default) that `viewport`
 * currently shows, without scrolling it: a focus handoff must never move the content the user is looking at.
 * Reports whether anything took focus.
 */
export const focusVisibleOperable = (
  viewport: HTMLElement | null,
  { edge = 'first', selector = OPERABLE }: { edge?: 'first' | 'last'; selector?: string } = {}
): boolean => {
  if (!viewport) {
    return false;
  }

  const bounds = viewport.getBoundingClientRect();
  const candidates = [...viewport.querySelectorAll<HTMLElement>(selector)];

  if (edge === 'last') {
    candidates.reverse();
  }

  const target = candidates.find((element) => {
    const rect = element.getBoundingClientRect();

    return rect.height > 0 && rect.bottom > bounds.top && rect.top < bounds.bottom;
  });

  target?.focus({ preventScroll: true });

  return target !== undefined;
};

/**
 * The owner's persistent polite live region. Screen readers announce changes inside a region that already exists,
 * not one inserted with its text, so non-blocking failures report here rather than through the notices showing them.
 */
export const GalleryAnnouncer = ({ message }: { message: string }) => (
  <VisuallyHidden aria-atomic="true" aria-live="polite" role="status">
    {message}
  </VisuallyHidden>
);

/**
 * A compact, non-blocking failure beside content that still belongs to the current scope. Its owner announces it
 * through `GalleryAnnouncer`.
 */
export const GalleryLoadNotice = ({
  message,
  read,
  retryLabel,
  onFocusLost,
  ...rowProps
}: StackProps & {
  message: string;
  read: GalleryReadState;
  retryLabel: string;
  onFocusLost?: () => void;
}) => (
  <HStack gap="2" justify="space-between" minW="0" {...rowProps}>
    <HStack gap="1.5" minW="0">
      <Icon as={TriangleAlertIcon} boxSize="3" color="fg.error" flexShrink={0} />
      <Text color="fg.muted" fontSize="xs" truncate>
        {message}
      </Text>
    </HStack>
    <RetryButton
      aria-label={retryLabel}
      isRetrying={read.isRetrying}
      size="xs"
      variant="ghost"
      onFocusLost={onFocusLost}
      onRetry={read.retry}
    />
  </HStack>
);
