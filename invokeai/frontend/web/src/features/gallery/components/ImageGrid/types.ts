import { RefObject } from 'react';
import { ListRange, VirtuosoGridHandle } from 'react-virtuoso';

export type VirtuosoGalleryContext = {
  virtuosoRef: RefObject<VirtuosoGridHandle>;
  rootRef: RefObject<HTMLDivElement>;
  rangeRef: RefObject<ListRange>;
};
