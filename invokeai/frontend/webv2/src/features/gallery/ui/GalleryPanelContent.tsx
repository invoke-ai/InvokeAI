import { Stack } from '@chakra-ui/react';

import { GalleryImageGrid } from './GalleryImageGrid';
import { GalleryToolbar } from './GalleryToolbar';

/**
 * The center region floats its chrome over the work surface and publishes the
 * clearance as `--wb-center-chrome-inset`. The gallery toolbar is top-anchored
 * and full width, so it adds that clearance to its own padding; every other
 * placement leaves the variable undefined and resolves to the plain padding.
 */
export const GalleryPanelContent = ({ layout }: { layout: 'stacked' | 'wide' }) => {
  const padding = layout === 'stacked' ? '2' : '3';

  return (
    <Stack
      gap="3"
      p={padding}
      paddingTop={`calc(var(--chakra-spacing-${padding}) + var(--wb-center-chrome-inset, 0px))`}
      h="full"
      maxW="full"
      minH="0"
      minW="0"
      w="full"
    >
      <GalleryToolbar layout={layout} />
      <GalleryImageGrid layout={layout} />
    </Stack>
  );
};
