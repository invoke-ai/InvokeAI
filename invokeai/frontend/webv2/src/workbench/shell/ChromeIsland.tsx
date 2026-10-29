import type { ReactNode } from 'react';

import { Flex, HStack } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';

import { CUE_WIDTH, HIDDEN_SCROLLBAR_CSS, OverflowCue, ROW_PADDING_X, useOverflowEdges } from './overflowEdges';

/**
 * The narrowest island that still shows `controlWidth` of its row's start clear of the end cue: border, start padding,
 * the control, and the cue.
 */
export const getChromeIslandMinWidth = (controlWidth: string): string =>
  `calc(${controlWidth} + {spacing.${ROW_PADDING_X}} + ${CUE_WIDTH} + 2px)`;

/**
 * A floating island of center chrome. Its row scrolls sideways, without a scrollbar, once the island reaches the width
 * its parent allows: by wheel, swipe, or focus, which brings a focused control clear of the cues. Each end hiding
 * content shows a fade into the island's background and a button that scrolls to that end.
 */
export const ChromeIsland = ({
  children,
  flexShrink = 1,
  maxW,
}: {
  children: ReactNode;
  flexShrink?: 0 | 1;
  maxW?: string;
}) => {
  const { t } = useTranslation();
  const { attach, hiddenEdges, scrollTo } = useOverflowEdges();

  return (
    <Flex
      align="center"
      bg="bg.subtle"
      borderColor="border.subtle"
      borderWidth="1px"
      data-center-chrome-island=""
      flexShrink={flexShrink}
      h="8"
      maxW={maxW}
      minW="0"
      pointerEvents="auto"
      position="relative"
      rounded="md"
      shadow="sm"
    >
      {/* Labels truncate down to their own minimum widths first; past those the row overflows and scrolls. */}
      <HStack
        ref={attach}
        css={HIDDEN_SCROLLBAR_CSS}
        flex="1"
        gap="0.5"
        // Reaches 2px past the island's border above and below: a scroller clips, and a focused control needs room for
        // its ring.
        h="calc(100% + 4px)"
        // Contains the controls' own z-indices, so the cues after the row always paint and take the pointer above them.
        isolation="isolate"
        minW="0"
        overflowX="auto"
        overflowY="hidden"
        // A swipe on the row scrolls it and nothing behind it, even at either end.
        overscrollBehaviorX="contain"
        px={ROW_PADDING_X}
        // Focus scrolls a control clear of the cue covering that end, not merely onto the island's edge.
        scrollPaddingInline={CUE_WIDTH}
        touchAction="pan-x"
      >
        {children}
      </HStack>
      {hiddenEdges.start ? <OverflowCue edge="start" label={t('common.scrollToStart')} onScrollTo={scrollTo} /> : null}
      {hiddenEdges.end ? <OverflowCue edge="end" label={t('common.scrollToEnd')} onScrollTo={scrollTo} /> : null}
    </Flex>
  );
};
