import type { GalleryImage } from '@features/gallery/core/types';

import { Box, Flex, Image, SimpleGrid, Skeleton, Text } from '@chakra-ui/react';
import { ALL_READABLE_BOARDS_ID, listPaletteImages } from '@features/gallery/data/backend';
import { getGalleryImageThumbnailUrl } from '@features/gallery/data/imageUrls';
import { useMountEffect } from '@platform/react/useMountEffect';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * The last few images this account produced, across every project.
 *
 * The gallery's own "recent images" is per-project widget state, so a
 * cross-project strip has to read the backend directly. It uses the same
 * cancellable read the command palette does, which is the entry point built
 * for exactly this kind of deferred cross-feature surface.
 */

const OUTPUT_COUNT = 8;
const GRID_COLUMNS = { base: 4, md: 8 } as const;

export const RecentOutputs = () => {
  const { t } = useTranslation();
  const [images, setImages] = useState<GalleryImage[] | null>(null);

  useMountEffect(() => {
    const owner = captureAccountScope();

    void listPaletteImages({
      boardId: ALL_READABLE_BOARDS_ID,
      galleryView: 'images',
      limit: OUTPUT_COUNT,
      searchTerm: '',
      signal: owner.signal,
    })
      .then((page) => {
        if (isAccountScopeCurrent(owner)) {
          setImages(page.images);
        }
      })
      .catch(() => {
        if (isAccountScopeCurrent(owner)) {
          setImages([]);
        }
      });
  });

  if (images !== null && images.length === 0) {
    return null;
  }

  return (
    <Flex direction="column" gap="3">
      <Text fontSize="xs" fontWeight="700">
        {t('launchpad.home.recentOutputs')}
      </Text>
      <SimpleGrid columns={GRID_COLUMNS} gap="2">
        {images === null
          ? Array.from({ length: OUTPUT_COUNT }, (_value, index) => (
              <Skeleton aspectRatio={1} key={index} rounded="md" />
            ))
          : images.map((image) => <RecentOutput image={image} key={image.imageName} />)}
      </SimpleGrid>
    </Flex>
  );
};

const RecentOutput = ({ image }: { image: GalleryImage }) => (
  <Box aspectRatio={1} bg="bg.muted" overflow="hidden" rounded="md">
    <Image alt="" h="full" objectFit="cover" src={getGalleryImageThumbnailUrl(image.imageName)} w="full" />
  </Box>
);
