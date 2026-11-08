import type { GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryMediaUsageCheck, GalleryMediaUsageReport } from '@features/gallery/mediaUsage';
import type { ReactNode } from 'react';

import { Box, Checkbox, Flex, Image, Stack, Text } from '@chakra-ui/react';
import { formatGalleryMediaCounts, toGalleryItemKey } from '@features/gallery/contracts';
import { GalleryMediaUsagePanel } from '@features/gallery/react';
import { galleryImageUrls, galleryVideoUrls } from '@features/gallery/utility';
import { useMountEffect } from '@platform/react/useMountEffect';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { Dialog } from '@platform/ui/Dialog';
import { toaster } from '@platform/ui/toaster';
import { patchWorkbenchPreferences, useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useCallback, useId, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

export interface DeletionConfirmationOptions {
  /**
   * The request may offer "Don't ask me again": only for one item the user can see. A batch, or an item off screen,
   * always asks, so the opt-out never deletes what the user is not looking at.
   */
  canSkipFutureConfirmations?: boolean;
  /** Where focus goes when the dialog closes; without it, back to the control that opened it. */
  returnFocus?: () => HTMLElement | null;
  /**
   * Where the items are still used, checked as the dialog opens: it shows the check running, then its report, so the
   * user sees what the deletion breaks. Aborted when the dialog closes. The dialog's Retry runs it again.
   */
  checkUsage?: (signal: AbortSignal) => Promise<GalleryMediaUsageReport>;
  /** A report the caller already has, shown at once instead of first running `checkUsage`. */
  usage?: GalleryMediaUsageReport;
}

export type RequestDeletionConfirmation = (
  itemRefs: readonly GalleryItemRef[],
  executeDeletion: () => Promise<void>,
  options?: DeletionConfirmationOptions
) => Promise<void>;

interface PendingDeletion {
  canSkipFutureConfirmations: boolean;
  checkUsage?: (signal: AbortSignal) => Promise<GalleryMediaUsageReport>;
  executeDeletion: () => Promise<void>;
  itemRefs: readonly GalleryItemRef[];
  resolve: () => void;
  /** Stops the running usage check: replaced by a retry, aborted once the request settles. */
  usageCheckController: AbortController;
}

/** The usage check of one request: a later request's check never shows in an earlier one's dialog, or the reverse. */
interface PendingUsageCheck {
  check: GalleryMediaUsageCheck;
  pending: PendingDeletion;
}

const NO_ITEMS: readonly GalleryItemRef[] = [];
/** Tiles in the dialog's one row of thumbnails: all of a small deletion, else one fewer and a count of the rest. */
export const DELETION_PREVIEW_SLOTS = 8;

const countMedia = (t: Parameters<typeof formatGalleryMediaCounts>[0], itemRefs: readonly GalleryItemRef[]): string =>
  formatGalleryMediaCounts(t, {
    images: itemRefs.filter((ref) => ref.kind === 'image').length,
    videos: itemRefs.filter((ref) => ref.kind === 'video').length,
  });

const getThumbnailUrl = (ref: GalleryItemRef): string =>
  ref.kind === 'video' ? galleryVideoUrls.thumbnail(ref.name) : galleryImageUrls.thumbnail(ref.name);

/** What the deletion takes, as one row of thumbnails: the first few, then how many more. */
const DeletionTargets = ({ itemRefs }: { itemRefs: readonly GalleryItemRef[] }) => {
  const { t } = useTranslation();
  const shown = itemRefs.length <= DELETION_PREVIEW_SLOTS ? itemRefs : itemRefs.slice(0, DELETION_PREVIEW_SLOTS - 1);
  const hiddenCount = itemRefs.length - shown.length;

  return (
    <Flex aria-label={t('widgets.gallery.deleteTargets')} as="ul" gap="1" role="list" wrap="nowrap">
      {shown.map((ref, index) => (
        <Box
          key={toGalleryItemKey(ref)}
          as="li"
          bg="bg.muted"
          borderRadius="sm"
          boxSize="10"
          flexShrink={0}
          overflow="hidden"
        >
          <Image
            alt={t(ref.kind === 'video' ? 'widgets.gallery.deleteTargetVideo' : 'widgets.gallery.deleteTargetImage', {
              index: index + 1,
            })}
            boxSize="full"
            draggable={false}
            objectFit="cover"
            src={getThumbnailUrl(ref)}
          />
        </Box>
      ))}
      {hiddenCount > 0 ? (
        <Flex
          align="center"
          aria-label={t('widgets.gallery.deleteTargetsMoreLabel', { count: hiddenCount })}
          as="li"
          bg="bg.muted"
          borderRadius="sm"
          boxSize="10"
          color="fg.muted"
          flexShrink={0}
          fontSize="md"
          fontVariantNumeric="tabular-nums"
          justify="center"
        >
          <span aria-hidden="true">{t('widgets.gallery.deleteTargetsMore', { count: hiddenCount })}</span>
        </Flex>
      ) : null}
    </Flex>
  );
};

export const useDeletionConfirmation = (): {
  dialog: ReactNode;
  requestDeletionConfirmation: RequestDeletionConfirmation;
} => {
  const { t } = useTranslation();
  const [pendingDeletion, setPendingDeletion] = useState<PendingDeletion | null>(null);
  const [usageCheck, setUsageCheck] = useState<PendingUsageCheck | null>(null);
  // Kept past the request: the dialog closes after the pending deletion clears, and resolves focus as it does.
  const [returnFocus, setReturnFocus] = useState<(() => HTMLElement | null) | null>(null);
  const pendingDeletionRef = useRef<PendingDeletion | null>(null);
  const deletionInFlightRef = useRef(false);
  // The same, for rendering: Retry stays in place but disabled while the confirmed deletion runs.
  const [isDeleting, setIsDeleting] = useState(false);
  // Read only when deletion is confirmed, so ticking it and then cancelling changes nothing.
  const [skipFutureConfirmations, setSkipFutureConfirmations] = useState(false);
  // With routine confirmations already off, the dialog only appears because the media is in use or the request
  // cannot skip it, which the opt-out does not cover.
  const confirmsRoutineDeletions = useWorkbenchPreferenceSelector((preferences) => preferences.confirmImageDeletion);
  const skipHintId = useId();

  /** Runs the request's check; its answer lands only in that request's dialog, and only from its latest run. */
  const runUsageCheck = useCallback((pending: PendingDeletion) => {
    const { checkUsage } = pending;

    if (!checkUsage) {
      return;
    }

    const controller = new AbortController();
    const settleCheck = (check: GalleryMediaUsageCheck) => {
      if (!controller.signal.aborted) {
        setUsageCheck((current) => (current?.pending === pending ? { check, pending } : current));
      }
    };

    pending.usageCheckController.abort();
    pending.usageCheckController = controller;
    // A retry keeps this request's last answer on screen while it runs; a first check has none.
    setUsageCheck((current) => {
      const previous =
        current?.pending !== pending
          ? undefined
          : current.check.status === 'checking'
            ? current.check.previous
            : current.check;

      return { check: previous ? { previous, status: 'checking' } : { status: 'checking' }, pending };
    });
    checkUsage(controller.signal).then(
      (report) => settleCheck({ report, status: 'ready' }),
      () => settleCheck({ status: 'failed' })
    );
  }, []);

  const settlePendingDeletion = useCallback(() => {
    const pending = pendingDeletionRef.current;

    if (!pending) {
      return;
    }

    pendingDeletionRef.current = null;
    deletionInFlightRef.current = false;
    setIsDeleting(false);
    pending.usageCheckController.abort();
    setPendingDeletion(null);
    pending.resolve();
  }, []);

  const requestDeletionConfirmation = useCallback<RequestDeletionConfirmation>(
    (
      itemRefs,
      executeDeletion,
      { canSkipFutureConfirmations = false, checkUsage, returnFocus: getReturnFocus, usage } = {}
    ) => {
      if (pendingDeletionRef.current) {
        // One dialog at a time. A second request is refused, and says so: with routine confirmations off it comes from
        // a check that found its media in use while another dialog waited, or while a confirmed deletion still runs,
        // long after the press, and would otherwise vanish without a word.
        toaster.create({
          title: t(
            deletionInFlightRef.current
              ? 'widgets.gallery.deleteBlockedByRunning'
              : 'widgets.gallery.deleteBlockedByPending',
            {
              count: itemRefs.length,
              media: countMedia(t, itemRefs),
            }
          ),
          type: 'warning',
        });
        return Promise.resolve();
      }

      return new Promise<void>((resolve) => {
        const pending: PendingDeletion = {
          canSkipFutureConfirmations,
          checkUsage,
          executeDeletion,
          itemRefs: [...itemRefs],
          resolve,
          usageCheckController: new AbortController(),
        };

        pendingDeletionRef.current = pending;
        setSkipFutureConfirmations(false);
        setPendingDeletion(pending);
        setReturnFocus(() => getReturnFocus ?? null);

        if (usage) {
          setUsageCheck({ check: { report: usage, status: 'ready' }, pending });
        } else if (checkUsage) {
          runUsageCheck(pending);
        } else {
          setUsageCheck(null);
        }
      });
    },
    [runUsageCheck, t]
  );

  const handleRetryUsageCheck = useCallback(() => {
    const pending = pendingDeletionRef.current;

    if (pending && !deletionInFlightRef.current) {
      runUsageCheck(pending);
    }
  }, [runUsageCheck]);

  const handleConfirm = useCallback(async () => {
    const pending = pendingDeletionRef.current;

    if (!pending || deletionInFlightRef.current) {
      return;
    }

    deletionInFlightRef.current = true;
    setIsDeleting(true);

    if (skipFutureConfirmations && pending.canSkipFutureConfirmations) {
      void patchWorkbenchPreferences({ confirmImageDeletion: false });
    }

    // Deleting anyway while the check runs: its answer no longer matters, and the panel stops saying it is checking.
    // A retry's earlier answer stays; a first check leaves nothing to show.
    pending.usageCheckController.abort();
    setUsageCheck((current) =>
      current?.pending !== pending || current.check.status !== 'checking'
        ? current
        : current.check.previous
          ? { check: current.check.previous, pending }
          : null
    );

    try {
      await pending.executeDeletion();
    } finally {
      if (pendingDeletionRef.current === pending) {
        pendingDeletionRef.current = null;
        deletionInFlightRef.current = false;
        setIsDeleting(false);
        setPendingDeletion(null);
        pending.resolve();
      }
    }
  }, [skipFutureConfirmations]);

  const handleSkipChange = useCallback((details: { checked: boolean | 'indeterminate' }) => {
    setSkipFutureConfirmations(details.checked === true);
  }, []);

  useMountEffect(() => settlePendingDeletion);

  const itemCount = pendingDeletion?.itemRefs.length ?? 0;
  const isImageOnly = pendingDeletion?.itemRefs.every((item) => item.kind === 'image') ?? false;
  const titleKey = isImageOnly ? 'widgets.gallery.deleteImagesConfirmTitle' : 'widgets.gallery.deleteItemsConfirmTitle';
  const bodyKey = isImageOnly ? 'widgets.gallery.deleteImagesConfirmBody' : 'widgets.gallery.deleteItemsConfirmBody';
  // Kept past the request like the rest of the body, so the closing dialog does not drop its report mid-animation.
  const shownUsageCheck = usageCheck?.check ?? null;
  const canRetryUsageCheck = usageCheck !== null && usageCheck.pending.checkUsage !== undefined;
  const itemRefs = pendingDeletion?.itemRefs ?? NO_ITEMS;
  const countSummary = countMedia(t, itemRefs);
  const canSkipFutureConfirmations = confirmsRoutineDeletions && pendingDeletion?.canSkipFutureConfirmations === true;

  return {
    dialog: (
      <ConfirmDialog
        body={
          <>
            <Dialog.Description color="fg" fontSize="md" lineHeight="inherit">
              {t(bodyKey, { count: itemCount })}
            </Dialog.Description>
            <Stack gap="1.5">
              <Text fontSize="md" fontWeight="semibold">
                {t('widgets.gallery.deleteTargetsSummary', { media: countSummary })}
              </Text>
              <DeletionTargets itemRefs={itemRefs} />
            </Stack>
            {shownUsageCheck ? (
              <GalleryMediaUsagePanel
                check={shownUsageCheck}
                // Also while the dialog closes: the request it would inform is settled.
                isRetryDisabled={isDeleting || pendingDeletion === null}
                onRetry={canRetryUsageCheck ? handleRetryUsageCheck : undefined}
              />
            ) : null}
            {canSkipFutureConfirmations ? (
              <Stack gap="1" pt="1">
                <Checkbox.Root checked={skipFutureConfirmations} size="sm" onCheckedChange={handleSkipChange}>
                  <Checkbox.HiddenInput aria-describedby={skipHintId} />
                  <Checkbox.Control>
                    <Checkbox.Indicator />
                  </Checkbox.Control>
                  <Checkbox.Label>{t('widgets.gallery.deleteConfirmSkip')}</Checkbox.Label>
                </Checkbox.Root>
                <Text color="fg.muted" fontSize="sm" id={skipHintId} ps="4.5">
                  {t('widgets.gallery.deleteConfirmSkipHint')}
                </Text>
              </Stack>
            ) : null}
          </>
        }
        confirmLabel={t('widgets.gallery.deleteConfirmLabel')}
        finalFocusEl={returnFocus ?? undefined}
        isOpen={pendingDeletion !== null}
        // The usage report arrives after the dialog opens; anchored at the top, it grows the dialog downward.
        placement="top"
        title={t(titleKey, { count: itemCount })}
        onClose={settlePendingDeletion}
        onConfirm={handleConfirm}
      />
    ),
    requestDeletionConfirmation,
  };
};
