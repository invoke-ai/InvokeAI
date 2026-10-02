import type { GalleryImage, GalleryItem } from '@features/gallery';
import type { ImageActions } from '@workbench/image-actions';

import { chakra, HStack, Icon, Popover, Portal, Text, VisuallyHidden } from '@chakra-ui/react';
import { formatGalleryVideoDuration, toGalleryItemKey, toGalleryItemRef } from '@features/gallery/contracts';
import { imageIndexAvailabilityOptions } from '@features/gallery/queries';
import { useAuthSession } from '@features/identity';
import { IconButton } from '@platform/ui/Button';
import { PopoverContent } from '@platform/ui/Popover';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { useQuery } from '@tanstack/react-query';
import { InfoIcon } from 'lucide-react';
import { useCallback, useMemo, type ComponentProps, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import type { PreviewItemPosition } from './previewHeaderStore';

import { PreviewDetails } from './PreviewMetadataPanel';

type InteractOutsideHandler = NonNullable<ComponentProps<typeof Popover.Root>['onInteractOutside']>;

const HeaderSeparator = () => (
  <Text color="fg.subtle" flexShrink={0} fontSize="2xs">
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
    <Text color="fg.muted" data-preview-image-tags flex="1 1 0" fontSize="2xs" minW="0" truncate>
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
 * Bound Details to the media stage so long metadata scrolls without covering the filmstrip; images offer parsed
 * details alongside raw payload tabs.
 */
export const PreviewDetailsPopover = ({
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
  const ids = useTooltipTriggerIds();
  const positioning = useMemo(
    () => ({
      boundary: () => stageElement ?? document.body,
      flip: false,
      overflowPadding: 8,
      placement: 'bottom-end' as const,
    }),
    [stageElement]
  );
  const handleOpenChange = useCallback(({ open }: { open: boolean }) => onOpenChange(open), [onOpenChange]);
  // Keep Details open while stage/filmstrip selection changes and preserve Preview keyboard navigation.
  const handleInteractOutside = useCallback<InteractOutsideHandler>(
    (event) => {
      const widget = stageElement?.closest('[role="region"]');
      const target = event.detail.originalEvent.target;

      if (widget && target instanceof Node && widget.contains(target)) {
        event.preventDefault();
      }
    },
    [stageElement]
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

  return (
    <Popover.Root
      autoFocus={false}
      ids={ids}
      lazyMount
      open={isOpen}
      positioning={positioning}
      unmountOnExit
      onInteractOutside={handleInteractOutside}
      onOpenChange={handleOpenChange}
    >
      <Tooltip content={t('widgets.preview.details')} ids={ids}>
        <Popover.Trigger asChild>
          <IconButton
            aria-label={t('widgets.preview.details')}
            aria-pressed={isOpen}
            color={isOpen ? undefined : 'fg.muted'}
            size="2xs"
            variant={isOpen ? 'solid' : 'ghost'}
          >
            <Icon as={InfoIcon} boxSize="3.5" />
          </IconButton>
        </Popover.Trigger>
      </Tooltip>
      <Portal>
        <Popover.Positioner>
          <PopoverContent
            data-preview-details
            display="flex"
            flexDirection="column"
            maxH="var(--available-height)"
            showArrow={false}
            w="24rem"
          >
            <Popover.Body display="flex" flexDirection="column" gap="2" minH="0" p="2.5">
              <HStack gap="1" minW="0">
                {positionLabel === null ? null : (
                  <>
                    <Text color="fg.muted" flexShrink={0} fontSize="2xs" fontVariantNumeric="tabular-nums">
                      {positionLabel}
                    </Text>
                    <HeaderSeparator />
                  </>
                )}
                <Text color="fg.muted" fontSize="2xs" fontVariantNumeric="tabular-nums" truncate>
                  {mediaLabel}
                </Text>
                <PreviewImageTags accountEpoch={accountEpoch} item={item} />
              </HStack>
              <PreviewDetails accountEpoch={accountEpoch} actions={actions} image={image} item={item} />
            </Popover.Body>
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};
