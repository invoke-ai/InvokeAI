import type { MouseEvent } from 'react';

import { Box } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { ChevronLeftIcon, ChevronRightIcon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { PreviewSwipeNavigation } from './usePreviewSwipe';

/** Marks the arrows so the stage can reveal them on hover or focus within it. */
export const PREVIEW_NAV_ATTRIBUTE = 'data-preview-nav';

/**
 * The stage shows its arrows while the pointer is over it or focus is within it. Hidden arrows take no pointer events,
 * so a click at the stage edge still reaches the image. The fade is opacity alone, and none under reduced motion.
 */
export const PREVIEW_NAV_STAGE_CSS = {
  [`& [${PREVIEW_NAV_ATTRIBUTE}]`]: {
    opacity: 0,
    pointerEvents: 'none',
    transitionDuration: '120ms',
    transitionProperty: 'opacity',
    transitionTimingFunction: 'ease-out',
    '@media (prefers-reduced-motion: reduce)': { transitionDuration: '0ms' },
  },
  [`&:hover [${PREVIEW_NAV_ATTRIBUTE}], &:focus-within [${PREVIEW_NAV_ATTRIBUTE}]`]: {
    opacity: 1,
    pointerEvents: 'auto',
  },
} as const;

/** Keeps the Preview's focus where it is, so the arrow keys still step after a click. */
const keepFocus = (event: MouseEvent) => event.preventDefault();

/**
 * Previous and next over the stage: a pointer path to the same steps the arrow keys take. A side with nothing to step
 * to has no button. Out of the tab order, since the keys already reach both steps.
 */
export const PreviewNavButtons = ({
  isVideo,
  navigation,
}: {
  isVideo: boolean;
  navigation: PreviewSwipeNavigation;
}) => {
  const { t } = useTranslation();
  const [isPending, setIsPending] = useState(false);
  const { neighbors, onNavigate } = navigation;
  const step = useCallback(
    (offset: -1 | 1) => {
      setIsPending(true);
      void onNavigate(offset).finally(() => setIsPending(false));
    },
    [onNavigate]
  );
  const stepPrevious = useCallback(() => step(-1), [step]);
  const stepNext = useCallback(() => step(1), [step]);

  return (
    <>
      {neighbors.previous ? (
        <NavButton
          isDisabled={isPending}
          isVideo={isVideo}
          keyShortcut="ArrowLeft"
          label={t('widgets.preview.commands.previousItem')}
          side="start"
          onStep={stepPrevious}
        />
      ) : null}
      {neighbors.next ? (
        <NavButton
          isDisabled={isPending}
          isVideo={isVideo}
          keyShortcut="ArrowRight"
          label={t('widgets.preview.commands.nextItem')}
          side="end"
          onStep={stepNext}
        />
      ) : null}
    </>
  );
};

const NavButton = ({
  isDisabled,
  isVideo,
  keyShortcut,
  label,
  side,
  onStep,
}: {
  isDisabled: boolean;
  isVideo: boolean;
  keyShortcut: string;
  label: string;
  side: 'end' | 'start';
  onStep: () => void;
}) => (
  <Box
    {...{ [PREVIEW_NAV_ATTRIBUTE]: side }}
    insetInlineEnd={side === 'end' ? '2' : undefined}
    insetInlineStart={side === 'start' ? '2' : undefined}
    position="absolute"
    // A video's controls run along the bottom of the stage; center over what is left above them.
    top={isVideo ? 'calc(50% - 1.5rem)' : '50%'}
    transform="translateY(-50%)"
    zIndex="1"
  >
    <Tooltip content={label}>
      <IconButton
        aria-keyshortcuts={keyShortcut}
        aria-label={label}
        bg="bg.panel/80"
        disabled={isDisabled}
        rounded="full"
        size="sm"
        tabIndex={-1}
        variant="outline"
        onClick={onStep}
        onMouseDown={keepFocus}
      >
        {side === 'start' ? <ChevronLeftIcon /> : <ChevronRightIcon />}
      </IconButton>
    </Tooltip>
  </Box>
);
