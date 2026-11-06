import type { GalleryImage, GalleryItem } from '@features/gallery';
import type { ImageActions, ImageContextMenuVariant } from '@workbench/image-actions';

import { Icon, HStack, Menu, Portal } from '@chakra-ui/react';
import { galleryImageItemToGalleryImage, isGalleryImageItem, toGalleryItemRef } from '@features/gallery/contracts';
import { canCopyImage } from '@platform/browser/clipboard';
import { Button, IconButton } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { releaseFocusOnRemoval } from '@workbench/focusRegions';
import {
  ImageContinueGroup,
  ImageUseGroups,
  useImageRecallCapabilities,
  VideoUseGroups,
} from '@workbench/image-actions';
import {
  ChevronDownIcon,
  CopyIcon,
  EllipsisVerticalIcon,
  ImagesIcon,
  PencilIcon,
  SquareArrowOutUpRightIcon,
  StarIcon,
  type LucideIcon,
} from 'lucide-react';
import { useCallback, useMemo, useState, type ComponentProps, type MouseEvent, type Ref } from 'react';
import { useTranslation } from 'react-i18next';

import type { PreviewDensity } from './previewDensity';

const MENU_POSITIONING = { placement: 'bottom-start' } as const;
const STAR_LABEL_KEYS = {
  image: { off: 'widgets.preview.starImage', on: 'widgets.preview.unstarImage' },
  video: { off: 'widgets.preview.starVideo', on: 'widgets.preview.unstarVideo' },
} as const;

/**
 * The selection's actions in the Preview header. An image gets Edit (every way to continue working with it) and Use
 * (it as an input, or its settings); a video gets Use alone. ⋮ then keeps only housekeeping: labelled at full density,
 * icon-only at compact, where compare moves into ⋮ too. Minimal density keeps star and the full ⋮ alone.
 */
export const PreviewActionStrip = ({
  actions,
  density,
  getRegion,
  isVideoFrameCopyAvailable = false,
  item,
  onCopyCurrentFrame,
  onOpenMenu,
}: {
  actions: ImageActions;
  density: PreviewDensity;
  /** Preview's focusable region, which takes focus from the ⋮ when a later render removes it. */
  getRegion?: () => HTMLElement | null;
  isVideoFrameCopyAvailable?: boolean;
  item: GalleryItem;
  onCopyCurrentFrame?: () => void;
  /** Opens the view's image context menu at viewport coordinates, leaving out what the strip already offers. */
  onOpenMenu: ((x: number, y: number, variant: ImageContextMenuVariant) => void) | null;
}) => {
  const { t } = useTranslation();
  const image = isGalleryImageItem(item) ? galleryImageItemToGalleryImage(item) : null;
  const toggleStar = useCallback(
    () => void actions.setItemsStarred([toGalleryItemRef(item)], !item.starred),
    [actions, item]
  );
  const selectForCompare = useCallback(() => image && actions.selectForCompare(image), [actions, image]);
  const copyCurrentFrame = useCallback(() => onCopyCurrentFrame?.(), [onCopyCurrentFrame]);
  const hasItemMenus = density !== 'minimal';
  const openMenu = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      const rect = event.currentTarget.getBoundingClientRect();

      onOpenMenu?.(rect.left, rect.bottom + 4, hasItemMenus ? 'housekeeping' : 'full');
    },
    [hasItemMenus, onOpenMenu]
  );
  const starLabel = t(STAR_LABEL_KEYS[item.kind][item.starred ? 'on' : 'off']);
  const starButton = (
    <Tooltip content={starLabel}>
      <IconButton aria-label={starLabel} color="fg.muted" size="sm" variant="ghost" onClick={toggleStar}>
        <Icon as={StarIcon} boxSize="3.5" fill={item.starred ? 'currentColor' : 'none'} />
      </IconButton>
    </Tooltip>
  );
  // Focus comes back to the ⋮ when its menu closes, and the strip goes with the selection.
  const menuButtonRef = useCallback(
    (node: HTMLButtonElement | null) => (node && getRegion ? releaseFocusOnRemoval(node, getRegion) : undefined),
    [getRegion]
  );
  const menuButton = onOpenMenu ? (
    <StripIconButton
      ref={menuButtonRef}
      icon={EllipsisVerticalIcon}
      label={t('widgets.preview.imageActions')}
      onClick={openMenu}
    />
  ) : null;

  if (density === 'compact') {
    return (
      <HStack flexShrink={0} gap="0.5">
        {image ? <EditMenu actions={actions} image={image} isIconOnly /> : null}
        <UseMenu actions={actions} isIconOnly item={item} />
        {starButton}
        {menuButton}
      </HStack>
    );
  }

  if (density !== 'full') {
    return (
      <HStack flexShrink={0} gap="0.5">
        {starButton}
        {menuButton}
      </HStack>
    );
  }

  return (
    <HStack flexShrink={0} gap="0.5">
      {/* Lead with the labelled next steps; videos have no Edit or compare, matching context-menu eligibility. */}
      {image ? <EditMenu actions={actions} image={image} /> : null}
      <UseMenu actions={actions} item={item} />
      {image ? (
        <StripIconButton icon={ImagesIcon} label={t('widgets.preview.selectForCompare')} onClick={selectForCompare} />
      ) : null}
      {starButton}
      {item.kind === 'video' && onCopyCurrentFrame ? (
        <StripIconButton
          // Without a clipboard (plain HTTP) the control says why; a frame still loading only disables it.
          disabled={!isVideoFrameCopyAvailable}
          icon={CopyIcon}
          label={t('widgets.preview.copyCurrentFrame')}
          unavailableReason={canCopyImage() ? undefined : t('common.copyUnavailable')}
          onClick={copyCurrentFrame}
        />
      ) : null}
      {menuButton}
    </HStack>
  );
};

/**
 * The item an open menu acts on, held from the moment it opens, as the ⋮ menu holds its target: a result that
 * Auto-switch brings in meanwhile must not take over the items, nor reload what they offer.
 */
const useHeldWhileOpen = <T,>(item: T) => {
  const [heldItem, setHeldItem] = useState<T | null>(null);
  const handleOpenChange = useCallback(({ open }: { open: boolean }) => setHeldItem(open ? item : null), [item]);

  return { handleOpenChange, menuItem: heldItem ?? item };
};

interface ItemMenuProps {
  actions: ImageActions;
  image: GalleryImage;
  /** Compact chrome: the icon and chevron alone, named by the tooltip and accessible name. */
  isIconOnly?: boolean;
}

/** The header's ghost menu trigger: icon, text and chevron, or the icon and chevron alone where width is short. */
const ItemMenuTrigger = ({
  icon,
  isIconOnly,
  label,
  text,
  ...triggerProps
}: {
  icon: LucideIcon;
  isIconOnly: boolean;
  label: string;
  text: string;
} & ComponentProps<'button'>) => (
  <Button
    {...triggerProps}
    aria-label={label}
    color="fg.muted"
    gap={isIconOnly ? '0' : undefined}
    minW={isIconOnly ? '0' : undefined}
    px={isIconOnly ? '0.5' : undefined}
    size="sm"
    variant="ghost"
  >
    <Icon as={icon} boxSize="3.5" />
    {isIconOnly ? null : text}
    {/* The size recipe sets every glyph to 14px; the icon-only chevron stays small, so the trigger stays narrow. */}
    {isIconOnly ? <Icon as={ChevronDownIcon} boxSize="2.5" /> : <ChevronDownIcon size={12} />}
  </Button>
);

/** Every way to continue working with the image, Edit on Canvas first. */
const EditMenu = ({ actions, image, isIconOnly = false }: ItemMenuProps) => {
  const { t } = useTranslation();
  const label = t('widgets.preview.editThisImage');
  const ids = useTooltipTriggerIds();
  const { handleOpenChange, menuItem: menuImage } = useHeldWhileOpen(image);
  const images = useMemo(() => [menuImage], [menuImage]);

  return (
    <Menu.Root ids={ids} lazyMount positioning={MENU_POSITIONING} unmountOnExit onOpenChange={handleOpenChange}>
      <Tooltip content={label} ids={ids}>
        <Menu.Trigger asChild>
          <ItemMenuTrigger
            icon={PencilIcon}
            isIconOnly={isIconOnly}
            label={isIconOnly ? label : t('common.edit')}
            text={t('common.edit')}
          />
        </Menu.Trigger>
      </Tooltip>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="16rem">
            <ImageContinueGroup actions={actions} images={images} presentation="guided" />
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

/**
 * What to take from the item: it as an input, or its recorded settings; for an image, with the viewer keys that do
 * the same.
 */
const UseMenu = ({
  actions,
  isIconOnly = false,
  item,
}: Omit<ItemMenuProps, 'image'> & {
  item: GalleryItem;
}) => {
  const { t } = useTranslation();
  const ids = useTooltipTriggerIds();
  const { handleOpenChange, menuItem } = useHeldWhileOpen(item);
  // Named for the held item, so an open menu keeps its name when Auto-switch brings in the other kind.
  const label = t(menuItem.kind === 'video' ? 'widgets.preview.useThisVideo' : 'widgets.preview.useInGenerate');

  return (
    <Menu.Root ids={ids} lazyMount positioning={MENU_POSITIONING} unmountOnExit onOpenChange={handleOpenChange}>
      <Tooltip content={label} ids={ids}>
        <Menu.Trigger asChild>
          <ItemMenuTrigger
            icon={SquareArrowOutUpRightIcon}
            isIconOnly={isIconOnly}
            label={label}
            text={t('widgets.preview.use')}
          />
        </Menu.Trigger>
      </Tooltip>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="16rem">
            {isGalleryImageItem(menuItem) ? (
              <UseMenuItems actions={actions} image={galleryImageItemToGalleryImage(menuItem)} />
            ) : (
              <VideoUseGroups actions={actions} item={menuItem} presentation="guided" />
            )}
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

/** Mounted with the open menu, so capabilities load on open rather than on every selection. */
const UseMenuItems = ({ actions, image }: { actions: ImageActions; image: GalleryImage }) => {
  const { capabilities, isLoading } = useImageRecallCapabilities(actions, image);

  return (
    <ImageUseGroups
      actions={actions}
      capabilities={capabilities}
      image={image}
      isLoadingCapabilities={isLoading}
      presentation="guided"
    />
  );
};

const StripIconButton = ({
  disabled,
  icon,
  label,
  ref,
  unavailableReason,
  onClick,
}: {
  disabled?: boolean;
  icon: LucideIcon;
  label: string;
  ref?: Ref<HTMLButtonElement>;
  unavailableReason?: string;
  onClick: (event: MouseEvent<HTMLButtonElement>) => void;
}) => (
  <IconButton
    ref={ref}
    aria-label={label}
    color="fg.muted"
    disabled={disabled}
    size="sm"
    tooltip={label}
    unavailableReason={unavailableReason}
    variant="ghost"
    onClick={onClick}
  >
    <Icon as={icon} boxSize="3.5" />
  </IconButton>
);
