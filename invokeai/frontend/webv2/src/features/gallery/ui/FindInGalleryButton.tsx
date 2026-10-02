import type { IconButtonProps } from '@platform/ui/Button';

import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { CrosshairIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

/** Share the reveal icon, accessible name, and tooltip across media thumbnails; callers own presentation. */
export type FindInGalleryButtonProps = Omit<
  IconButtonProps,
  'aria-label' | 'aria-labelledby' | 'children' | 'onClick'
> & {
  /** Include the media filename so assistive technology distinguishes simultaneous reveal buttons. */
  name?: string;
  onFind: () => void;
};

export const FindInGalleryButton = ({ name, onFind, ...buttonProps }: FindInGalleryButtonProps) => {
  const { t } = useTranslation();
  // One string for both: a tooltip that is not part of the accessible name
  // leaves speech input with nothing to match when the user says what they see.
  const label =
    name === undefined ? t('widgets.gallery.findInGallery') : t('widgets.gallery.findNamedInGallery', { name });

  return (
    <Tooltip content={label}>
      <IconButton aria-label={label} size="2xs" {...buttonProps} onClick={onFind}>
        <CrosshairIcon />
      </IconButton>
    </Tooltip>
  );
};

// Reveal on keyboard focus as well as hover; opacity and pointer-events preserve tab access that display:none
// would remove.
const REVEALED = { opacity: 1, pointerEvents: 'auto' } as const;

/**
 * Host with className="group" and positioned layout. Inset the overlay by the focus ring's reach to prevent
 * thumbnail clipping.
 */
export const FindInGalleryThumbnailButton = (buttonProps: FindInGalleryButtonProps) => (
  <FindInGalleryButton
    bottom="1"
    colorPalette="gray"
    insetInlineEnd="1"
    opacity={0}
    pointerEvents="none"
    position="absolute"
    transition="opacity var(--wb-motion-duration-medium) ease"
    variant="solid"
    zIndex="1"
    _groupFocusWithin={REVEALED}
    _groupHover={REVEALED}
    {...buttonProps}
  />
);
