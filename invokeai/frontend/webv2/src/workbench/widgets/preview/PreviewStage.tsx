import type { Ref } from 'react';

import { Box, Flex, type BoxProps, type FlexProps, type SystemStyleObject } from '@chakra-ui/react';

import type { PreviewDensity } from './previewDensity';

export const getFittedFrameCss = (width: number, height: number): SystemStyleObject => ({
  // The cue gallery thumbnails give when a touch hold arms their drag; only a draggable frame is ever armed.
  '&[data-drag-armed=true]': { filter: 'saturate(0)' },
  aspectRatio: `${width} / ${height}`,
  height: 'auto',
  maxHeight: '100%',
  maxWidth: '100%',
  width: `min(100cqw, calc(100cqh * ${width / height}))`,
});

/** Room for the centre's chrome islands; only `CenterArea` sets the variable, so `0px` keeps others unpadded. */
const CENTER_CHROME_INSET = 'var(--wb-center-chrome-inset, 0px)';

/** Chrome clearance on the content box; the larger of inset and stage padding is the top padding. */
const getStagePaddingTop = (padding: string | undefined): string =>
  padding === undefined ? CENTER_CHROME_INSET : `max(var(--chakra-spacing-${padding}), ${CENTER_CHROME_INSET})`;

/** Stage padding by density, shared by the single item's stage and the comparison stage so their geometry matches. */
export const getPreviewStagePadding = (density: PreviewDensity): string => (density === 'full' ? '6' : '3');

/** Use parent fill for inset widget bodies and flex fill for framed stages. */
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
    containerType="size"
    justify="center"
    layerStyle="dotGrid"
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
