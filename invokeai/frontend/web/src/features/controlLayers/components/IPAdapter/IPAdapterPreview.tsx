import type { SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Flex,
  Image,
  Popover,
  PopoverAnchor,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  Portal,
} from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { useDisclosure } from 'common/hooks/useBoolean';
import { useFilterableOutsideClick } from 'common/hooks/useFilterableOutsideClick';
import { IPAdapterSettings } from 'features/controlLayers/components/IPAdapter/IPAdapterSettings';
import { useRefImageIdContext } from 'features/controlLayers/contexts/RefImageIdContext';
import { selectRefImageEntityOrThrow, selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import type { ImageWithDims } from 'features/controlLayers/store/types';
import { memo, useMemo, useRef } from 'react';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

const sx: SystemStyleObject = {
  opacity: 0.5,
  _hover: {
    opacity: 1,
  },
  "&[data-is-open='true']": {
    opacity: 1,
    pointerEvents: 'none',
  },
  transitionProperty: 'opacity',
  transitionDuration: '0.2s',
};

export const RefImagePreview = memo(() => {
  const id = useRefImageIdContext();
  const ref = useRef<HTMLDivElement>(null);
  const disclosure = useDisclosure(false);
  const selectEntity = useMemo(
    () =>
      createSelector(selectRefImagesSlice, (refImages) =>
        selectRefImageEntityOrThrow(refImages, id, 'RefImagePreview')
      ),
    [id]
  );
  const entity = useAppSelector(selectEntity);
  useFilterableOutsideClick({ ref, handler: disclosure.close });

  return (
    <Popover isLazy lazyBehavior="unmount" isOpen={disclosure.isOpen} closeOnBlur={false}>
      <PopoverAnchor>
        <Flex role="button" w={16} h={16} sx={sx} onClick={disclosure.open} data-is-open={disclosure.isOpen}>
          <Thumbnail image={entity.ipAdapter.image} />
        </Flex>
      </PopoverAnchor>
      <Portal>
        <PopoverContent ref={ref}>
          <PopoverArrow />
          <PopoverBody>
            <IPAdapterSettings />
          </PopoverBody>
        </PopoverContent>
      </Portal>
    </Popover>
  );
});
RefImagePreview.displayName = 'RefImagePreview';

const Thumbnail = memo(({ image }: { image: ImageWithDims | null }) => {
  const { data: imageDTO } = useGetImageDTOQuery(image?.image_name ?? skipToken);
  return <Image src={imageDTO?.thumbnail_url} objectFit="contain" maxW="full" maxH="full" />;
});
Thumbnail.displayName = 'Thumbnail';
