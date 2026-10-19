import { HStack, Icon, Spinner, Text, VisuallyHidden, type StackProps } from '@chakra-ui/react';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button } from '@platform/ui/Button';
import { EmptyState } from '@platform/ui/EmptyState';
import { TriangleAlertIcon } from 'lucide-react';
import { useCallback, useRef, type ComponentProps } from 'react';
import { useTranslation } from 'react-i18next';

import type { GalleryReadState } from './galleryStateView';

const ERROR_ICON = <Icon as={TriangleAlertIcon} />;

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

type GalleryRetryButtonProps = Omit<ComponentProps<typeof Button>, 'aria-label' | 'children' | 'onClick'> & {
  /** Names what is retried; it must begin with the visible "Retry". */
  'aria-label': string;
  /** Leads with an alert icon, for a Retry that stands alone without a message beside it. */
  flagsFailure?: boolean;
  read: Pick<GalleryReadState, 'isRetrying' | 'retry'>;
  /**
   * Where focus goes when a successful retry removes this button while it held focus. Without it focus would
   * fall to the document body; a user who already moved focus elsewhere keeps it.
   */
  onFocusLost?: () => void;
};

/** Stays mounted and focused while busy: `aria-disabled` rather than `disabled`, which would drop focus. */
export const GalleryRetryButton = ({ flagsFailure = false, onFocusLost, read, ...props }: GalleryRetryButtonProps) => {
  const { t } = useTranslation();
  const nodeRef = useRef<HTMLButtonElement | null>(null);
  const { isRetrying, retry } = read;
  const attach = useCallback(
    (node: HTMLButtonElement | null) => {
      const previous = nodeRef.current;

      nodeRef.current = node;

      if (node || !onFocusLost || !previous?.contains(document.activeElement)) {
        return;
      }

      // Detaching runs before the node leaves the document; hand focus on once the replacement has committed.
      queueMicrotask(() => {
        if (document.activeElement === null || document.activeElement === document.body) {
          onFocusLost();
        }
      });
    },
    [onFocusLost]
  );
  const handleClick = useCallback(() => {
    if (!isRetrying) {
      void retry();
    }
  }, [isRetrying, retry]);

  return (
    <Button
      ref={attach}
      aria-busy={isRetrying || undefined}
      aria-disabled={isRetrying || undefined}
      flexShrink={0}
      variant="outline"
      {...props}
      onClick={handleClick}
    >
      {isRetrying ? <Spinner boxSize="3" /> : flagsFailure ? <Icon as={TriangleAlertIcon} boxSize="3" /> : null}
      {t('common.retry')}
    </Button>
  );
};

/** Takes the place of content a failed load left with nothing to show for the current scope. */
export const GalleryLoadErrorState = ({
  read,
  retryLabel,
  title,
  onFocusLost,
}: {
  read: GalleryReadState;
  retryLabel: string;
  title: string;
  onFocusLost?: () => void;
}) => (
  <EmptyState
    aria-busy={read.isRetrying || undefined}
    danger
    description={read.error ? getApiErrorMessage(read.error, '') : null}
    flex="1"
    icon={ERROR_ICON}
    py="6"
    role="alert"
    title={title}
  >
    <GalleryRetryButton aria-label={retryLabel} read={read} size="sm" onFocusLost={onFocusLost} />
  </EmptyState>
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
    <GalleryRetryButton aria-label={retryLabel} read={read} size="xs" variant="ghost" onFocusLost={onFocusLost} />
  </HStack>
);
