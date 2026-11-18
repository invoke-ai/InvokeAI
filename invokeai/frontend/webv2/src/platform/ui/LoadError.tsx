import { HStack, Icon, Spinner, Stack, Text, type StackProps } from '@chakra-ui/react';
import { TriangleAlertIcon } from 'lucide-react';
import { useCallback, useRef, useState, type ComponentProps } from 'react';
import { useTranslation } from 'react-i18next';

import { Button } from './Button';
import { EmptyState } from './EmptyState';

const ERROR_ICON = <Icon as={TriangleAlertIcon} />;

export type RetryButtonProps = Omit<ComponentProps<typeof Button>, 'children' | 'onClick'> & {
  isRetrying: boolean;
  /** Return the attempt's promise: the button stays busy, and refuses another press, until it settles. */
  onRetry: () => unknown;
  /** Leads with an alert icon, for a Retry that stands alone without a message beside it. */
  flagsFailure?: boolean;
  /**
   * Where focus goes when a successful retry removes this button while it held focus. Without it focus would fall to
   * the document body; a user who already moved focus elsewhere keeps it.
   */
  onFocusLost?: () => void;
};

/**
 * "Retry" for a failed read. It stays mounted and focused while the attempt runs: `aria-disabled` and a spinner
 * rather than `disabled`, which would drop focus. An `aria-label` naming what is retried must begin with "Retry".
 */
export const RetryButton = ({ flagsFailure = false, isRetrying, onFocusLost, onRetry, ...props }: RetryButtonProps) => {
  const { t } = useTranslation();
  const nodeRef = useRef<HTMLButtonElement | null>(null);
  // Set on the press itself, so a double click cannot start a second attempt before `isRetrying` reports the first.
  const inFlightRef = useRef(false);
  const [isStarting, setIsStarting] = useState(false);
  const isBusy = isRetrying || isStarting;
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
    if (isRetrying || inFlightRef.current) {
      return;
    }

    const settle = () => {
      inFlightRef.current = false;
      setIsStarting(false);
    };

    inFlightRef.current = true;
    setIsStarting(true);
    // The owner reports a failed attempt; this only needs to know when it is over.
    try {
      void Promise.resolve(onRetry()).then(settle, settle);
    } catch {
      settle();
    }
  }, [isRetrying, onRetry]);

  return (
    <Button
      ref={attach}
      aria-busy={isBusy || undefined}
      aria-disabled={isBusy || undefined}
      flexShrink={0}
      variant="outline"
      {...props}
      onClick={handleClick}
    >
      {isBusy ? <Spinner boxSize="3" /> : flagsFailure ? <Icon as={TriangleAlertIcon} boxSize="3" /> : null}
      {t('common.retry')}
    </Button>
  );
};

export type LoadErrorProps = Omit<StackProps, 'align' | 'children' | 'role' | 'size' | 'title'> & {
  /** What could not be read, worded for the user. */
  message: string;
  /** The server's reason, when it gave one. */
  detail?: string | null;
  isRetrying: boolean;
  /** `state` takes the place of content the failure left with nothing to show; `inline` sits beside content still shown. */
  layout: 'inline' | 'state';
  onRetry: () => unknown;
  /** The inline layout's scale: `sm` inside dense panels, `md` beside page-sized content such as the Launchpad's. */
  size?: 'md' | 'sm';
  /** Names what is retried, for a screen reader; it must begin with the visible "Retry". */
  retryLabel?: string;
  /** See `RetryButton`. */
  onFocusLost?: () => void;
};

/** A read that failed, never shown as an empty list: an alert with what failed, why when known, and a retry. */
export const LoadError = ({
  detail,
  isRetrying,
  layout,
  message,
  onFocusLost,
  onRetry,
  retryLabel,
  size = 'sm',
  ...rootProps
}: LoadErrorProps) => {
  const isState = layout === 'state';
  const textSize = size === 'md' ? 'md' : 'xs';
  const retry = (
    <RetryButton
      aria-label={retryLabel}
      isRetrying={isRetrying}
      size={isState || size === 'md' ? 'sm' : 'xs'}
      onFocusLost={onFocusLost}
      onRetry={onRetry}
    />
  );

  if (isState) {
    return (
      <EmptyState
        aria-busy={isRetrying || undefined}
        danger
        description={detail}
        icon={ERROR_ICON}
        role="alert"
        title={message}
        {...rootProps}
      >
        {retry}
      </EmptyState>
    );
  }

  return (
    <HStack alignItems="flex-start" aria-busy={isRetrying || undefined} gap="1.5" role="alert" {...rootProps}>
      <Icon as={TriangleAlertIcon} boxSize={size === 'md' ? '4' : '3'} color="fg.error" flexShrink={0} mt="px" />
      <Stack flex="1" gap="0.5" minW="0">
        <Text color="fg.error" fontSize={textSize} textWrap="pretty">
          {message}
        </Text>
        {detail ? (
          <Text color="fg.muted" fontSize={textSize} wordBreak="break-word">
            {detail}
          </Text>
        ) : null}
      </Stack>
      {retry}
    </HStack>
  );
};
