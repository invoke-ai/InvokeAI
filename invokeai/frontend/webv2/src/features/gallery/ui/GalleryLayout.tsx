import { Box } from '@chakra-ui/react';

import type { GalleryWidgetProps } from './GalleryUiContext';

import { useGalleryLayout } from './galleryDensity';
import { GalleryStackedLayout } from './GalleryStackedLayout';
import { GalleryWideLayout } from './GalleryWideLayout';

export const GalleryLayout = ({ region }: { region: GalleryWidgetProps['region'] }) => {
  const { layout, rootRef } = useGalleryLayout(region);

  return (
    <Box ref={rootRef} h="full" maxW="full" minH="0" minW="0" w="full">
      {layout === 'wide' ? <GalleryWideLayout /> : <GalleryStackedLayout />}
    </Box>
  );
};
