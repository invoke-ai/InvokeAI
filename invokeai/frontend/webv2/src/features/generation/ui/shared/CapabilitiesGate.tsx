import type { ReactNode } from 'react';

import { Box, HStack, Spinner, Stack, Text } from '@chakra-ui/react';
import {
  ensureArchitectureCapabilitiesLoaded,
  getArchitectureCapabilitiesSnapshot,
  useArchitectureCapabilitiesSelector,
} from '@features/generation/data/architectureCapabilitiesStore';
import { focusIfUnclaimed } from '@platform/react/focusIfUnclaimed';
import { Button } from '@platform/ui/Button';
import { useCallback, useRef, useState } from 'react';

export interface CapabilitiesGateProps {
  /**
   * Whether the form can render. Until it can, the gate shows the load, and a retry once it fails.
   * Generate passes a finished load, which is also when its values resolve; Upscale passes the table
   * itself (a nonzero registry revision), which is what its sync and validation read. In the app the
   * two agree: the table arrives only with a successful load and leaves only with the account.
   */
  isReady: boolean;
  /** Translated copy, so each widget keeps its own strings. */
  labels: { loadFailed: string; loading: string; retry: string };
  children: ReactNode;
}

/**
 * Holds back a form whose defaults and validation come from the served architecture table, so no
 * edit made before it loads can persist a fallback in place of a served default.
 */
export const CapabilitiesGate = ({ children, isReady, labels }: CapabilitiesGateProps) => {
  const status = useArchitectureCapabilitiesSelector((snapshot) => snapshot.status);
  const error = useArchitectureCapabilitiesSelector((snapshot) => snapshot.error);
  const [hasRequestedRetry, setHasRequestedRetry] = useState(false);
  // Keep the failure surface mounted during retry to preserve button focus.
  const isRetrying = hasRequestedRetry && status === 'loading';

  // Set by the click, consumed when the form the retry revealed mounts.
  const focusHandoffPending = useRef(false);

  const retry = useCallback(() => {
    if (isRetrying) {
      return;
    }

    focusHandoffPending.current = true;
    setHasRequestedRetry(true);
    // Only the initiating retry owns focus; unrelated later reloads must not take it.
    void ensureArchitectureCapabilitiesLoaded().then(() => {
      focusHandoffPending.current &&= getArchitectureCapabilitiesSnapshot().status === 'loaded';
      setHasRequestedRetry(false);
    });
  }, [isRetrying]);

  const handOverFocus = useCallback((element: HTMLDivElement | null) => {
    if (element && focusHandoffPending.current) {
      focusHandoffPending.current = false;
      focusIfUnclaimed(element);
    }
  }, []);

  if (!isReady) {
    if (status === 'error' || isRetrying) {
      return (
        <Stack aria-busy={isRetrying} aria-live="polite" gap="2" justify="center" minH="8rem" p="1" role="alert">
          <Text color="fg.error" fontSize="xs" textWrap="pretty">
            {labels.loadFailed}
          </Text>
          {error ? (
            <Text color="fg.muted" fontSize="xs" textWrap="pretty">
              {error}
            </Text>
          ) : null}
          {/* `aria-disabled` rather than `disabled`: a disabled button drops the focus it holds. */}
          <Button
            alignSelf="flex-start"
            aria-busy={isRetrying}
            aria-disabled={isRetrying}
            variant="outline"
            onClick={retry}
          >
            {isRetrying ? <Spinner /> : null}
            {labels.retry}
          </Button>
        </Stack>
      );
    }

    return (
      <HStack
        align="center"
        aria-atomic="true"
        aria-busy="true"
        aria-live="polite"
        color="fg.muted"
        gap="1.5"
        justify="center"
        minH="8rem"
        p="1"
        role="status"
      >
        <Spinner />
        <Text fontSize="xs">{labels.loading}</Text>
      </HStack>
    );
  }

  // A successful retry unmounts the button that held focus; hand it to the form the retry revealed.
  return <Box ref={handOverFocus}>{children}</Box>;
};
