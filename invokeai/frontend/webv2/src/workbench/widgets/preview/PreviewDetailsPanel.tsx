import type { GalleryImage, GalleryItem } from '@features/gallery';
import type { ImageActions } from '@workbench/image-actions';

import { Box, chakra, HStack, Text, VisuallyHidden } from '@chakra-ui/react';
import { formatGalleryVideoDuration, toGalleryItemKey, toGalleryItemRef } from '@features/gallery/contracts';
import { imageIndexAvailabilityOptions } from '@features/gallery/queries';
import { useAuthSession } from '@features/identity';
import { ToggleIconButton } from '@platform/ui/Button';
import { useQuery } from '@tanstack/react-query';
import { useWorkbenchFocus } from '@workbench/focusRegions';
import { isWidgetRegion } from '@workbench/layoutContracts';
import { InfoIcon } from 'lucide-react';
import { useCallback, useId, useRef, type KeyboardEvent, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { useTranslation } from 'react-i18next';

import type { PreviewItemPosition } from './previewHeaderStore';

import { PreviewDetails } from './PreviewMetadataPanel';

const HeaderSeparator = () => (
  <Text color="fg.subtle" flexShrink={0} fontSize="xs">
    ·
  </Text>
);

/**
 * The item's image-map tags, best first. The label cache is loaded on open, keeping it out of the editor's initial
 * graph, and answers repeat opens without a request; it owns invalidation (vocabulary rebuilds, accounts), so this
 * query keeps nothing once closed.
 */
const PreviewImageTags = ({ accountEpoch, item }: { accountEpoch: number; item: GalleryItem }): ReactNode => {
  const { t } = useTranslation();
  const { data: indexAvailability } = useQuery(imageIndexAvailabilityOptions());
  const isIndexReady = indexAvailability?.state === 'ready';
  const { data: tags } = useQuery({
    enabled: isIndexReady,
    gcTime: 0,
    queryFn: async () => {
      const { getImageLabels } = await import('@workbench/image-map/imageLabelCache');
      const labels = await getImageLabels(toGalleryItemRef(item));

      return labels ? [labels.label, ...labels.alternates].slice(0, 3) : null;
    },
    queryKey: ['preview', 'image-tags', accountEpoch, toGalleryItemKey(item)],
    retry: false,
    staleTime: 0,
  });

  // Tags answered while the index was ready must not outlive it.
  if (!isIndexReady || !tags || tags.length === 0) {
    return null;
  }

  // A zero basis gives the tags only the width the header leaves over, so they truncate before anything else and,
  // with the separator inside them, vanish whole rather than leave a dangling one.
  return (
    <Text color="fg.muted" data-preview-image-tags flex="1 1 0" fontSize="xs" minW="0" truncate>
      <chakra.span aria-hidden="true" color="fg.subtle" marginInlineEnd="1">
        ·
      </chakra.span>
      <VisuallyHidden>{t('widgets.preview.imageTags')} </VisuallyHidden>
      {tags.map((tag, index) => (
        <chakra.span key={tag} fontWeight={index === 0 ? '600' : undefined}>
          {index > 0 ? ', ' : ''}
          {tag}
        </chakra.span>
      ))}
    </Text>
  );
};

/**
 * The centre region's inset already spans its chrome islands' top offset, their height and an equal gap below them, so
 * the panel sits as far under the islands as they sit under the region's top. Elsewhere the variable is unset and the
 * panel takes the islands' own offset from the stage's corner; `insetEnd` matches their end inset either way.
 */
const PANEL_TOP = 'var(--wb-center-chrome-inset, var(--chakra-spacing-2))';
const PANEL_MAX_HEIGHT = `calc(100% - ${PANEL_TOP} - var(--chakra-spacing-2))`;

/**
 * Details as a panel pinned to the media stage's top-end corner, beside the header trigger and off the filmstrip;
 * images offer parsed details alongside raw payload tabs. It is not a popup: nothing outside closes it, and it listens
 * to no document events, so other surfaces keep their own Escape, pointer and focus handling. Only its trigger, or
 * Escape from within it, closes it. It follows the selection and re-anchors whenever the stage remounts.
 */
export const PreviewDetailsPanel = ({
  actions,
  image,
  isOpen,
  item,
  onOpenChange,
  position,
  stageElement,
}: {
  actions: ImageActions;
  image: GalleryImage | null;
  isOpen: boolean;
  item: GalleryItem;
  onOpenChange: (open: boolean) => void;
  position: PreviewItemPosition | null;
  stageElement: HTMLElement | null;
}) => {
  const { i18n, t } = useTranslation();
  const accountEpoch = useAuthSession().accountEpoch;
  const panelId = useId();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const label = t('widgets.preview.details');
  // React bubbles a portalled popup's keys (a menu opened inside) through here too, so only Escape pressed within the
  // panel's own DOM, and not already handled, closes it.
  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (
        event.key !== 'Escape' ||
        event.defaultPrevented ||
        event.nativeEvent.isComposing ||
        !(event.target instanceof Node && event.currentTarget.contains(event.target))
      ) {
        return;
      }
      event.preventDefault();
      onOpenChange(false);
    },
    [onOpenChange]
  );
  const focusMoves = useWorkbenchFocus();
  // However Details leaves (its trigger, Escape, the I hotkey, a selection that takes the header item, a stage that
  // remounts), focus inside it would fall to the body. It goes to the trigger instead, or, when the trigger left too, to
  // the Preview's own window or region. Read once removal is committed: a cleanup alone cannot tell removal apart from a
  // re-attached ref, and focus that already moved elsewhere stays put.
  const restoreFocusOnRemoval = useCallback(
    (element: HTMLDivElement | null) => {
      if (!element) {
        return;
      }
      return () => {
        const document = element.ownerDocument;
        if (!element.contains(document.activeElement)) {
          return;
        }
        const floatingId = element.closest<HTMLElement>('[data-floating-window]')?.dataset.floatingWindow;
        const region = element.closest<HTMLElement>('[data-focus-region]')?.dataset.focusRegion;
        queueMicrotask(() => {
          const active = document.activeElement;
          if (element.isConnected || (active !== null && active !== document.body)) {
            return;
          }
          const trigger = triggerRef.current;
          if (trigger?.isConnected) {
            trigger.focus();
          } else if (floatingId) {
            focusMoves.focusFloating(floatingId);
          } else if (isWidgetRegion(region)) {
            focusMoves.focusRegion(region, 'preview');
          }
        });
      };
    },
    [focusMoves]
  );
  const positionLabel =
    position === null
      ? null
      : position.isLoadingBoard
        ? t('widgets.preview.loadingBoard')
        : position.selectedIndex === -1
          ? t('widgets.preview.itemCount', { count: position.boardItemCount })
          : t('common.countOfTotal', { count: position.selectedIndex + 1, total: position.boardItemCount });
  const fps =
    item.kind === 'video' && item.fps !== undefined
      ? new Intl.NumberFormat(i18n.language, { maximumFractionDigits: 3 }).format(item.fps)
      : null;
  const mediaLabel = [
    `${item.width} × ${item.height}`,
    ...(item.kind === 'video'
      ? [t('widgets.preview.videoDuration', { duration: formatGalleryVideoDuration(item.durationSeconds) })]
      : []),
    ...(fps === null ? [] : [t('widgets.preview.framesPerSecond', { count: fps })]),
  ].join(' · ');
  const panel =
    isOpen && stageElement
      ? createPortal(
          // A named region rather than a group: a pinned panel is a place to return to, so it is a landmark. It
          // follows the stage's media in DOM order, so Tab reaches it from the Preview without moving focus on open.
          <Box
            ref={restoreFocusOnRemoval}
            aria-label={label}
            animationDuration="faster"
            animationName="slide-from-top, fade-in"
            bg="bg.muted"
            borderColor="border.emphasized"
            borderRadius="l3"
            borderWidth="1px"
            boxShadow="lg"
            color="fg"
            data-preview-details
            display="flex"
            flexDirection="column"
            id={panelId}
            insetEnd="2"
            maxH={PANEL_MAX_HEIGHT}
            maxW="calc(100% - 1rem)"
            outline="0"
            position="absolute"
            role="region"
            textStyle="sm"
            top={PANEL_TOP}
            w="24rem"
            // Above the media and its step buttons, but within the stage: never over dialogs or menus.
            zIndex="docked"
            onKeyDown={handleKeyDown}
          >
            <Box display="flex" flex="1" flexDirection="column" gap="2" minH="0" p="2.5">
              <HStack gap="1" minW="0">
                {positionLabel === null ? null : (
                  <>
                    <Text color="fg.muted" flexShrink={0} fontSize="xs" fontVariantNumeric="tabular-nums">
                      {positionLabel}
                    </Text>
                    <HeaderSeparator />
                  </>
                )}
                <Text color="fg.muted" fontSize="xs" fontVariantNumeric="tabular-nums" truncate>
                  {mediaLabel}
                </Text>
                <PreviewImageTags accountEpoch={accountEpoch} item={item} />
              </HStack>
              <PreviewDetails accountEpoch={accountEpoch} actions={actions} image={image} item={item} />
            </Box>
          </Box>,
          stageElement
        )
      : null;

  return (
    <>
      <ToggleIconButton
        ref={triggerRef}
        aria-controls={panel ? panelId : undefined}
        checked={isOpen}
        icon={InfoIcon}
        label={label}
        onCheckedChange={onOpenChange}
      />
      {panel}
    </>
  );
};
