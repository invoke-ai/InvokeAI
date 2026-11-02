import type { GalleryMediaUsageCheck } from '@features/gallery/core/mediaUsage';

import { HStack, Spinner, Stack, Text } from '@chakra-ui/react';
import { lazy, Suspense, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

// The report renders only once a check settles, so its code loads then (the check itself has usually loaded it).
const GalleryMediaUsageOutcome = lazy(() =>
  import('./GalleryMediaUsageSection').then((module) => ({ default: module.GalleryMediaUsageOutcome }))
);

/**
 * Where the media a deletion takes is still used, for the item and board deletion dialogs alike. Kept small and eager
 * on purpose: the live region exists as the dialog opens, so the report is announced when it arrives, and the checking
 * state shows at once, before any of the report's code has loaded.
 */
export const GalleryMediaUsagePanel = ({
  check,
  onRetry,
}: {
  check: GalleryMediaUsageCheck;
  /** Runs the check again after it failed or could not reach the server. */
  onRetry?: () => void;
}) => {
  const panelRef = useRef<HTMLDivElement | null>(null);
  // Retry is replaced by the checking state, and its control may not come back. Focus moves first to the panel itself,
  // which stays mounted through every state: it never falls out of the dialog, and the live region reads the new state.
  const handleRetry = useCallback(() => {
    panelRef.current?.focus();
    onRetry?.();
  }, [onRetry]);

  const settled = check.status === 'checking' ? check.previous : check;

  return (
    <Stack
      ref={panelRef}
      aria-busy={check.status === 'checking'}
      aria-live="polite"
      focusVisibleRing="outside"
      gap="1"
      // Room between the ring and the text, given back by the margin so focus moves nothing.
      m="-1.5"
      p="1.5"
      rounded="l2"
      // Focusable only by script, as Retry's stand-in: no tab stop of its own.
      tabIndex={-1}
    >
      {settled ? (
        // A settled answer never shows as "checking" while its code loads: nothing, for a frame, then the answer. A
        // retry keeps it on screen, the checking state standing where its warning and Retry were.
        <Suspense fallback={null}>
          <GalleryMediaUsageOutcome
            check={settled}
            isRechecking={check.status === 'checking'}
            onRetry={onRetry ? handleRetry : undefined}
          />
        </Suspense>
      ) : (
        <GalleryMediaUsageChecking />
      )}
    </Stack>
  );
};

/** The check running: shared by the first check and a retry, which shows it in place of the warning it answers. */
export const GalleryMediaUsageChecking = () => {
  const { t } = useTranslation();

  return (
    <HStack color="fg.muted" gap="2">
      <Spinner aria-hidden="true" boxSize="3" />
      <Text fontSize="md">{t('widgets.gallery.deleteInUse.checking')}</Text>
    </HStack>
  );
};
