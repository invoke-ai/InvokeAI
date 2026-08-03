import { Box } from '@chakra-ui/react';

import type { GalleryWidgetProps } from './GalleryUiContext';

import { useGalleryLayout } from './galleryDensity';
import { GalleryStackedLayout } from './GalleryStackedLayout';
import { GalleryWideLayout } from './GalleryWideLayout';

/**
 * The single place in the gallery that knows how much room it has. Everything
 * below is layout-blind: the two shells arrange one identical set of slot
 * components, so swapping between them is a rendering decision rather than a
 * behavioural one.
 */
export const GalleryLayout = ({ region }: { region: GalleryWidgetProps['region'] }) => {
  const { layout, rootRef } = useGalleryLayout(region);

  return (
    <Box ref={rootRef} h="full" maxW="full" minH="0" minW="0" w="full">
      {layout === 'wide' ? <GalleryWideLayout /> : <GalleryStackedLayout />}
    </Box>
  );
};
