import { Stack } from '@chakra-ui/react';

import { GalleryImageGrid } from './GalleryImageGrid';
import { GalleryToolbar } from './GalleryToolbar';

export const GalleryPanelContent = ({ layout }: { layout: 'stacked' | 'wide' }) => (
  <Stack gap="3" p={layout === 'stacked' ? '2' : '3'} h="full" maxW="full" minH="0" minW="0" w="full">
    <GalleryToolbar layout={layout} />
    <GalleryImageGrid layout={layout} />
  </Stack>
);
