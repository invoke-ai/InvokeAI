import type { GalleryItemRef } from '@features/gallery/contracts';
import type { ReactNode } from 'react';

import { Box, Checkbox, Flex, Image, List, Stack, Text } from '@chakra-ui/react';
import { formatGalleryMediaCounts, toGalleryItemKey } from '@features/gallery/contracts';
import { galleryImageUrls, galleryVideoUrls } from '@features/gallery/utility';
import { useMountEffect } from '@platform/react/useMountEffect';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { patchWorkbenchPreferences, useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useCallback, useId, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { MediaUsage } from './mediaUsage';

export interface DeletionConfirmationOptions {
  /**
   * The request may offer "Don't ask me again": only for one item the user can see. A batch, or an item off screen,
   * always asks, so the opt-out never deletes what the user is not looking at.
   */
  canSkipFutureConfirmations?: boolean;
  /** Where focus goes when the dialog closes; without it, back to the control that opened it. */
  returnFocus?: () => HTMLElement | null;
  /** Project surfaces that still use some of the items; listed so the user sees what the deletion breaks. */
  usage?: readonly MediaUsage[];
}

export type RequestDeletionConfirmation = (
  itemRefs: readonly GalleryItemRef[],
  executeDeletion: () => Promise<void>,
  options?: DeletionConfirmationOptions
) => Promise<void>;

interface PendingDeletion {
  canSkipFutureConfirmations: boolean;
  executeDeletion: () => Promise<void>;
  itemRefs: readonly GalleryItemRef[];
  resolve: () => void;
  usage: readonly MediaUsage[];
}

const NO_USAGE: readonly MediaUsage[] = [];
const NO_ITEMS: readonly GalleryItemRef[] = [];
/** Tiles in the dialog's one row of thumbnails: all of a small deletion, else one fewer and a count of the rest. */
export const DELETION_PREVIEW_SLOTS = 8;

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
  // Kept past the request: the dialog closes after the pending deletion clears, and resolves focus as it does.
  const [returnFocus, setReturnFocus] = useState<(() => HTMLElement | null) | null>(null);
  const pendingDeletionRef = useRef<PendingDeletion | null>(null);
  const deletionInFlightRef = useRef(false);
  // Read only when deletion is confirmed, so ticking it and then cancelling changes nothing.
  const [skipFutureConfirmations, setSkipFutureConfirmations] = useState(false);
  // With routine confirmations already off, the dialog only appears because the media is in use or the request
  // cannot skip it, which the opt-out does not cover.
  const confirmsRoutineDeletions = useWorkbenchPreferenceSelector((preferences) => preferences.confirmImageDeletion);
  const skipHintId = useId();
  const usageHeadingId = useId();

  const settlePendingDeletion = useCallback(() => {
    const pending = pendingDeletionRef.current;

    if (!pending) {
      return;
    }

    pendingDeletionRef.current = null;
    deletionInFlightRef.current = false;
    setPendingDeletion(null);
    pending.resolve();
  }, []);

  const requestDeletionConfirmation = useCallback<RequestDeletionConfirmation>(
    (itemRefs, executeDeletion, { canSkipFutureConfirmations = false, returnFocus: getReturnFocus, usage } = {}) => {
      if (pendingDeletionRef.current) {
        return Promise.resolve();
      }

      return new Promise<void>((resolve) => {
        const pending = {
          canSkipFutureConfirmations,
          executeDeletion,
          itemRefs: [...itemRefs],
          resolve,
          usage: usage ?? NO_USAGE,
        };

        pendingDeletionRef.current = pending;
        setSkipFutureConfirmations(false);
        setPendingDeletion(pending);
        setReturnFocus(() => getReturnFocus ?? null);
      });
    },
    []
  );

  const handleConfirm = useCallback(async () => {
    const pending = pendingDeletionRef.current;

    if (!pending || deletionInFlightRef.current) {
      return;
    }

    deletionInFlightRef.current = true;

    if (skipFutureConfirmations && pending.canSkipFutureConfirmations) {
      void patchWorkbenchPreferences({ confirmImageDeletion: false });
    }

    try {
      await pending.executeDeletion();
    } finally {
      if (pendingDeletionRef.current === pending) {
        pendingDeletionRef.current = null;
        deletionInFlightRef.current = false;
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
  const usage = pendingDeletion?.usage ?? NO_USAGE;
  const itemRefs = pendingDeletion?.itemRefs ?? NO_ITEMS;
  const countSummary = formatGalleryMediaCounts(t, {
    images: itemRefs.filter((ref) => ref.kind === 'image').length,
    videos: itemRefs.filter((ref) => ref.kind === 'video').length,
  });
  const canSkipFutureConfirmations = confirmsRoutineDeletions && pendingDeletion?.canSkipFutureConfirmations === true;

  return {
    dialog: (
      <ConfirmDialog
        body={
          <>
            <Text fontSize="md">{t(bodyKey, { count: itemCount })}</Text>
            <Stack gap="1.5">
              <Text fontSize="md" fontWeight="semibold">
                {t('widgets.gallery.deleteTargetsSummary', { media: countSummary })}
              </Text>
              <DeletionTargets itemRefs={itemRefs} />
            </Stack>
            {usage.length > 0 ? (
              <Stack aria-labelledby={usageHeadingId} as="section" gap="1" role="group">
                <Text fontWeight="semibold" id={usageHeadingId}>
                  {t('widgets.gallery.deleteInUse.heading')}
                </Text>
                <List.Root gap="0.5" ps="4">
                  {usage.map((entry, index) => (
                    // Captured when the dialog opens and never reordered; labels such as workflow names can repeat.
                    <List.Item key={index}>
                      {t(`widgets.gallery.deleteInUse.surface.${entry.surface}`, { name: entry.label })}
                      <Text as="span" color="fg.muted">
                        {' · '}
                        {t('widgets.gallery.deleteInUse.usedCount', { count: entry.names.length })}
                      </Text>
                    </List.Item>
                  ))}
                </List.Root>
                <Text color="fg.muted" fontSize="sm">
                  {t('widgets.gallery.deleteInUse.warning')}
                </Text>
              </Stack>
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
        title={t(titleKey, { count: itemCount })}
        onClose={settlePendingDeletion}
        onConfirm={handleConfirm}
      />
    ),
    requestDeletionConfirmation,
  };
};
