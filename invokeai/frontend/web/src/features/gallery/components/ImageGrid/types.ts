import type { RefObject } from 'react';
import type { ListRange, VirtuosoGridHandle } from 'react-virtuoso';

export type VirtuosoGalleryContext = {
  virtuosoRef: RefObject<VirtuosoGridHandle>;
  rootRef: RefObject<HTMLDivElement>;
  virtuosoRangeRef: RefObject<ListRange>;
};
