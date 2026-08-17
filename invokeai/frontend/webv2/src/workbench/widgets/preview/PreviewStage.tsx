import type { Ref } from 'react';

import { Box, Flex, type BoxProps, type FlexProps, type SystemStyleObject } from '@chakra-ui/react';

/**
 * The chrome every preview surface shares, so the image, video, live, and empty
 * branches stop carrying their own copies:
 *
 * - `PreviewStage` — the dot-grid floor. Centres content, is the size container
 *   `getFittedFrameCss` measures against, and reserves top clearance for the
 *   centre region's floating islands.
 * - `FittedFrame` — the media card: aspect-fitted, bordered, shadowed.
 */

export const previewGridCss = {
  backgroundImage: 'radial-gradient(circle, currentColor 1px, transparent 1.5px)',
  backgroundPosition: 'center',
  backgroundRepeat: 'repeat',
  backgroundSize: '24px 24px',
} as const;

export const getFittedFrameCss = (width: number, height: number): SystemStyleObject => ({
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
 * Top padding for a media stage: its own padding plus room for the islands.
 *
 * It belongs on the stage rather than on the widget root so the dot grid still
 * runs to every edge and passes *behind* the chrome — the same arrangement that
 * lets `paddingBottom` pass it behind the footer island. Because the stage is a
 * size container, `getFittedFrameCss` reads the shrunken content box and refits
 * the media with no further change.
 */
const getStagePaddingTop = (padding: string | undefined): string =>
  padding === undefined ? CENTER_CHROME_INSET : `calc(var(--chakra-spacing-${padding}) + ${CENTER_CHROME_INSET})`;

/**
 * `fill="parent"` is the inset arrangement (`h="full"`, used where the stage IS
 * the widget body); `fill="flex"` is the framed one (`flex="1" minH="0"
 * overflow="hidden"`, used where overlays float over the stage's lower edge).
 *
 * Both fills are positioned: absolutely-placed stage children (zoom badge,
 * future overlays) anchor to the stage itself, never to whatever happens to
 * contain it. (The old inset copy was static; the difference is unobservable
 * at its call sites and the uniform rule is the one worth keeping.)
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
    rounded="lg"
    {...props}
  />
);
