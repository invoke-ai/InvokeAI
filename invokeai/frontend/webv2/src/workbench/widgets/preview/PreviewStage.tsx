import type { Ref } from 'react';

import { Box, Flex, type BoxProps, type FlexProps, type SystemStyleObject } from '@chakra-ui/react';

/**
 * Share PreviewStage's fitted dot-grid surface and chrome clearance across media/live/empty states; FittedFrame
 * supplies bordered, shadowed media geometry.
 */

export const previewGridCss = {
  backgroundImage: 'radial-gradient(circle, currentColor 1px, transparent 1.5px)',
  backgroundPosition: 'center',
  backgroundRepeat: 'repeat',
  backgroundSize: '24px 24px',
} as const;

export const getFittedFrameCss = (width: number, height: number): SystemStyleObject => ({
  // The cue gallery thumbnails give when a touch hold arms their drag; only a draggable frame is ever armed.
  '&[data-drag-armed=true]': { filter: 'saturate(0)' },
  aspectRatio: `${width} / ${height}`,
  height: 'auto',
  maxHeight: '100%',
  maxWidth: '100%',
  width: `min(100cqw, calc(100cqh * ${width / height}))`,
});

/**
 * Room for the centre region's floating chrome islands. `CenterArea` publishes
 * the variable and nothing else does, so the `0px` fallback keeps the right
 * panel — which reserves a real header row instead — unpadded.
 */
const CENTER_CHROME_INSET = 'var(--wb-center-chrome-inset, 0px)';

/**
 * Reserve chrome clearance on the stage content box so fitting shrinks media while the dot grid continues behind
 * chrome.
 */
const getStagePaddingTop = (padding: string | undefined): string =>
  padding === undefined ? CENTER_CHROME_INSET : `calc(var(--chakra-spacing-${padding}) + ${CENTER_CHROME_INSET})`;

/**
 * Use parent fill for inset widget bodies and flex fill for framed stages. Both establish positioning for absolute
 * overlay children.
 */
export const PreviewStage = ({
  fill,
  padding,
  paddingBottom,
  ...props
}: Omit<FlexProps, 'padding' | 'paddingBottom'> & {
  fill: 'flex' | 'parent';
  padding?: string;
  paddingBottom?: string;
  ref?: Ref<HTMLDivElement>;
}) => (
  <Flex
    align="center"
    backgroundColor="bg.inset"
    color="fg.grid"
    containerType="size"
    css={previewGridCss}
    justify="center"
    p={padding}
    pb={paddingBottom}
    position="relative"
    pt={getStagePaddingTop(padding)}
    w="full"
    {...(fill === 'parent' ? { h: 'full' } : { flex: '1', minH: '0', overflow: 'hidden' })}
    {...props}
  />
);

export const FittedFrame = ({
  frameHeight,
  frameWidth,
  ...props
}: BoxProps & { frameHeight: number; frameWidth: number; ref?: Ref<HTMLDivElement> }) => (
  <Box
    borderColor="border.emphasized"
    borderWidth="1px"
    boxShadow="0 24px 80px rgba(0,0,0,0.42)"
    css={getFittedFrameCss(frameWidth, frameHeight)}
    overflow="hidden"
    position="relative"
    {...props}
  />
);
