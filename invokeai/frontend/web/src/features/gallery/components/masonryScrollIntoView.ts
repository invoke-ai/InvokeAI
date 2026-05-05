type MasonryMountedRange = {
  endIndex: number;
  startIndex: number;
};

type MasonryScrollDirection = 'down' | 'up' | null;

type GetMasonryScrollDirectionArg = {
  mountedRange: MasonryMountedRange | null;
  previousIndex?: number;
  targetIndex: number;
};

type ScrollMasonryImageIntoViewArg = {
  imageName: string;
  previousIndex?: number;
  rootEl: HTMLDivElement;
  targetIndex: number;
};

const MAX_MASONRY_SCROLL_RETRIES = 8;

export const getMasonryScrollDirection = ({
  mountedRange,
  previousIndex,
  targetIndex,
}: GetMasonryScrollDirectionArg): MasonryScrollDirection => {
  if (mountedRange) {
    if (targetIndex < mountedRange.startIndex) {
      return 'up';
    }
    if (targetIndex > mountedRange.endIndex) {
      return 'down';
    }
    return null;
  }

  if (previousIndex === undefined) {
    return null;
  }

  if (targetIndex < previousIndex) {
    return 'up';
  }
  if (targetIndex > previousIndex) {
    return 'down';
  }
  return null;
};

const getMountedMasonryRange = (rootEl: HTMLDivElement): MasonryMountedRange | null => {
  let startIndex = Number.POSITIVE_INFINITY;
  let endIndex = Number.NEGATIVE_INFINITY;

  for (const el of rootEl.querySelectorAll<HTMLElement>('[data-absolute-index]')) {
    const index = Number.parseInt(el.dataset.absoluteIndex ?? '', 10);
    if (!Number.isFinite(index)) {
      continue;
    }
    startIndex = Math.min(startIndex, index);
    endIndex = Math.max(endIndex, index);
  }

  if (startIndex === Number.POSITIVE_INFINITY || endIndex === Number.NEGATIVE_INFINITY) {
    return null;
  }

  return { endIndex, startIndex };
};

const getMasonryScroller = (rootEl: HTMLDivElement): HTMLElement | null => {
  return rootEl.querySelector<HTMLElement>('[data-testid="virtuoso-scroller"]');
};

const getMountedMasonryItem = (rootEl: HTMLDivElement, imageName: string): HTMLElement | null => {
  for (const el of rootEl.querySelectorAll<HTMLElement>('[data-item-id]')) {
    if (el.dataset.itemId !== imageName) {
      continue;
    }
    return el.closest<HTMLElement>('[data-absolute-index]');
  }

  return null;
};

const scrollScrollerByViewport = (scroller: HTMLElement, direction: Exclude<MasonryScrollDirection, null>) => {
  const amount = Math.max(1, scroller.clientHeight) * (direction === 'down' ? 1 : -1);

  if (typeof scroller.scrollBy === 'function') {
    scroller.scrollBy({ behavior: 'auto', top: amount });
  } else {
    scroller.scrollTop += amount;
  }
};

export const scrollMasonryImageIntoView = (arg: ScrollMasonryImageIntoViewArg): void => {
  const scroll = (attempt: number) => {
    const mountedItem = getMountedMasonryItem(arg.rootEl, arg.imageName);
    if (mountedItem) {
      mountedItem.scrollIntoView({ block: 'nearest', inline: 'nearest' });
      return;
    }

    if (attempt >= MAX_MASONRY_SCROLL_RETRIES) {
      return;
    }

    const scroller = getMasonryScroller(arg.rootEl);
    if (!scroller) {
      return;
    }

    const direction = getMasonryScrollDirection({
      mountedRange: getMountedMasonryRange(arg.rootEl),
      previousIndex: arg.previousIndex,
      targetIndex: arg.targetIndex,
    });

    if (!direction) {
      return;
    }

    scrollScrollerByViewport(scroller, direction);
    requestAnimationFrame(() => scroll(attempt + 1));
  };

  scroll(0);
};
