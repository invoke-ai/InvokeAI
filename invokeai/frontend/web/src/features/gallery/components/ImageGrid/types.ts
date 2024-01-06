import { atom } from 'nanostores';
import type { RefObject } from 'react';
import type { ListRange, VirtuosoGridHandle } from 'react-virtuoso';

export type VirtuosoGridRefs = {
  virtuosoRef?: RefObject<VirtuosoGridHandle>;
  rootRef?: RefObject<HTMLDivElement>;
  virtuosoRangeRef?: RefObject<ListRange>;
};

export const virtuosoGridRefs = atom<VirtuosoGridRefs>({});
