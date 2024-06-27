import { Flex } from '@invoke-ai/ui-library';

import { GalleryBulkSelect } from './GalleryBulkSelect';
import { GallerySort } from './GallerySort';

export const GalleryMenu = () => {
  return (
    <Flex alignItems="center" justifyContent="space-between">
      <GalleryBulkSelect />
      <GallerySort />
    </Flex>
  );
};
