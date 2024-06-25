import { Flex } from '@invoke-ai/ui-library';
import { GallerySort } from './GallerySort';
import { GalleryBulkSelect } from './GalleryBulkSelect';

export const GalleryMenu = () => {
  return (
    <Flex alignItems="center" justifyContent="space-between">
      <GalleryBulkSelect />
      <GallerySort />
    </Flex>
  );
};
