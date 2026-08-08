type MasonryMountedRange = {
  endIndex: number;
  startIndex: number;
};

type MasonryScrollDirection = 'down' | 'up' | null;

export type MasonryKeyboardNavigationDirection = 'down' | 'left' | 'right' | 'up';

type GetMasonryScrollDirectionArg = {
  mountedRange: MasonryMountedRange | null;
  previousIndex?: number;
  targetIndex: number;
};

type MasonryVisibilityRect = Pick<DOMRect, 'bottom' | 'left' | 'right' | 'top'>;

type MasonryKeyboardNavigationRect = MasonryVisibilityRect & Pick<DOMRect, 'height' | 'width'>;

type MasonryKeyboardNavigationItem = {
  imageName: string;
  index: number;
  rect: MasonryKeyboardNavigationRect;
};

type GetMasonryKeyboardNavigationTargetArg = {
  columnCount: number;
  currentImageName: string | null;
  currentIndex: number;
  direction: MasonryKeyboardNavigationDirection;
  imageNames: string[];
  mountedItems: MasonryKeyboardNavigationItem[];
};

type MasonryKeyboardNavigationTarget = {
  imageName: string;
  index: number;
};

type MasonryKeyboardCandidateScore = readonly [number, number, number, number];

type MasonrySelectedImageScrollState = {
  hasSelectionChangedSinceMount: boolean;
  initialSelectedImageName: string | null;
};

export type MasonryScrollTarget = {
  imageName: string;
  previousIndex?: number;
  targetIndex: number;
};

type GetMasonrySelectedImageScrollDecisionArg = {
  currentSelectedImageName: string | null;
  state: MasonrySelectedImageScrollState;
};

type GetMasonrySelectionScrollTargetArg = {
  currentImageName: string;
  pendingTarget: MasonryScrollTarget | null;
  targetIndex: number;
};

type GetMasonrySelectionAnchorDecisionArg = {
  isCurrentlyVisible: boolean;
  wasPreviouslyVisible: boolean;
};

type GetShouldHandleMasonryKeyboardRepeatArg = {
  currentImageName: string | null;
  isCurrentImageVisible: boolean;
  isRepeat: boolean;
};

type MasonryAnimationFrameCallback = (time: number) => void;

type CreateMasonrySelectionAnchorSchedulerArg = {
  cancelFrame?: (handle: number) => void;
  onSettled: () => void;
  requestFrame?: (callback: MasonryAnimationFrameCallback) => number;
};

type ScrollMasonryImageIntoViewArg = MasonryScrollTarget & {
  rootEl: HTMLDivElement;
};

const MAX_MASONRY_SCROLL_RETRIES = 8;
const MASONRY_KEYBOARD_NAVIGATION_EPSILON = 0.5;

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

export const getMasonrySelectedImageScrollDecision = ({
  currentSelectedImageName,
  state,
}: GetMasonrySelectedImageScrollDecisionArg): {
  nextState: MasonrySelectedImageScrollState;
  shouldScroll: boolean;
} => {
  const hasSelectionChangedSinceMount =
    state.hasSelectionChangedSinceMount || currentSelectedImageName !== state.initialSelectedImageName;

  return {
    nextState: {
      ...state,
      hasSelectionChangedSinceMount,
    },
    shouldScroll: currentSelectedImageName !== null && hasSelectionChangedSinceMount,
  };
};

export const getMasonrySelectionScrollTarget = ({
  currentImageName,
  pendingTarget,
  targetIndex,
}: GetMasonrySelectionScrollTargetArg): MasonryScrollTarget => ({
  imageName: currentImageName,
  previousIndex: pendingTarget?.imageName === currentImageName ? pendingTarget.previousIndex : undefined,
  targetIndex,
});

export const getMasonrySelectionAnchorDecision = ({
  isCurrentlyVisible,
  wasPreviouslyVisible,
}: GetMasonrySelectionAnchorDecisionArg) => ({
  nextWasVisible: isCurrentlyVisible,
  shouldAnchor: wasPreviouslyVisible && !isCurrentlyVisible,
});

export const getShouldHandleMasonryKeyboardRepeat = ({
  currentImageName,
  isCurrentImageVisible,
  isRepeat,
}: GetShouldHandleMasonryKeyboardRepeatArg) => !isRepeat || currentImageName === null || isCurrentImageVisible;

export const isMasonryItemRectVisible = ({
  itemRect,
  scrollerRect,
}: {
  itemRect: MasonryVisibilityRect;
  scrollerRect: MasonryVisibilityRect;
}) =>
  itemRect.bottom > scrollerRect.top &&
  itemRect.top < scrollerRect.bottom &&
  itemRect.right > scrollerRect.left &&
  itemRect.left < scrollerRect.right;

export const createMasonrySelectionAnchorScheduler = ({
  cancelFrame = cancelAnimationFrame,
  onSettled,
  requestFrame = requestAnimationFrame,
}: CreateMasonrySelectionAnchorSchedulerArg) => {
  let firstFrame: number | null = null;
  let secondFrame: number | null = null;

  const cancel = () => {
    if (firstFrame !== null) {
      cancelFrame(firstFrame);
      firstFrame = null;
    }
    if (secondFrame !== null) {
      cancelFrame(secondFrame);
      secondFrame = null;
    }
  };

  const schedule = () => {
    if (firstFrame !== null || secondFrame !== null) {
      return;
    }

    firstFrame = requestFrame(() => {
      firstFrame = null;
      secondFrame = requestFrame(() => {
        secondFrame = null;
        onSettled();
      });
    });
  };

  return { cancel, schedule };
};

const getRectCenterX = (rect: MasonryKeyboardNavigationRect) => rect.left + rect.width / 2;

const getRectCenterY = (rect: MasonryKeyboardNavigationRect) => rect.top + rect.height / 2;

const getRangeGap = (aStart: number, aEnd: number, bStart: number, bEnd: number) => {
  if (aEnd < bStart) {
    return bStart - aEnd;
  }
  if (bEnd < aStart) {
    return aStart - bEnd;
  }
  return 0;
};

const getMasonryKeyboardFallbackIndex = ({
  columnCount,
  currentIndex,
  direction,
  imageCount,
}: {
  columnCount: number;
  currentIndex: number;
  direction: MasonryKeyboardNavigationDirection;
  imageCount: number;
}) => {
  switch (direction) {
    case 'left':
      return Math.max(0, currentIndex - 1);
    case 'right':
      return Math.min(imageCount - 1, currentIndex + 1);
    case 'up':
      return Math.max(0, currentIndex - columnCount);
    case 'down':
      return Math.min(imageCount - 1, currentIndex + columnCount);
  }
};

const getMasonryKeyboardFallbackTarget = ({
  columnCount,
  currentIndex,
  direction,
  imageNames,
}: {
  columnCount: number;
  currentIndex: number;
  direction: MasonryKeyboardNavigationDirection;
  imageNames: string[];
}): MasonryKeyboardNavigationTarget | null => {
  if (imageNames.length === 0) {
    return null;
  }

  const index = getMasonryKeyboardFallbackIndex({
    columnCount,
    currentIndex,
    direction,
    imageCount: imageNames.length,
  });
  const imageName = imageNames[index];
  return imageName ? { imageName, index } : null;
};

const isMasonryKeyboardCandidateInDirection = (
  currentItem: MasonryKeyboardNavigationItem,
  candidate: MasonryKeyboardNavigationItem,
  direction: MasonryKeyboardNavigationDirection
) => {
  const currentCenterX = getRectCenterX(currentItem.rect);
  const currentCenterY = getRectCenterY(currentItem.rect);
  const candidateCenterX = getRectCenterX(candidate.rect);
  const candidateCenterY = getRectCenterY(candidate.rect);

  switch (direction) {
    case 'left':
      return candidateCenterX < currentCenterX - MASONRY_KEYBOARD_NAVIGATION_EPSILON;
    case 'right':
      return candidateCenterX > currentCenterX + MASONRY_KEYBOARD_NAVIGATION_EPSILON;
    case 'up':
      return candidateCenterY < currentCenterY - MASONRY_KEYBOARD_NAVIGATION_EPSILON;
    case 'down':
      return candidateCenterY > currentCenterY + MASONRY_KEYBOARD_NAVIGATION_EPSILON;
  }
};

const getMasonryKeyboardCandidateScore = (
  currentItem: MasonryKeyboardNavigationItem,
  candidate: MasonryKeyboardNavigationItem,
  direction: MasonryKeyboardNavigationDirection
): MasonryKeyboardCandidateScore => {
  const currentRect = currentItem.rect;
  const candidateRect = candidate.rect;
  const currentCenterX = getRectCenterX(currentRect);
  const currentCenterY = getRectCenterY(currentRect);
  const candidateCenterX = getRectCenterX(candidateRect);
  const candidateCenterY = getRectCenterY(candidateRect);
  const indexDistance = Math.abs(candidate.index - currentItem.index);

  if (direction === 'left' || direction === 'right') {
    const verticalGap = getRangeGap(currentRect.top, currentRect.bottom, candidateRect.top, candidateRect.bottom);
    const horizontalDistance =
      direction === 'left'
        ? Math.max(0, currentRect.left - candidateRect.right)
        : Math.max(0, candidateRect.left - currentRect.right);
    return [verticalGap, horizontalDistance, Math.abs(currentCenterY - candidateCenterY), indexDistance] as const;
  }

  const horizontalGap = getRangeGap(currentRect.left, currentRect.right, candidateRect.left, candidateRect.right);
  const verticalDistance =
    direction === 'up'
      ? Math.max(0, currentRect.top - candidateRect.bottom)
      : Math.max(0, candidateRect.top - currentRect.bottom);
  return [horizontalGap, verticalDistance, Math.abs(currentCenterX - candidateCenterX), indexDistance] as const;
};

const compareMasonryKeyboardCandidateScores = (a: MasonryKeyboardCandidateScore, b: MasonryKeyboardCandidateScore) => {
  if (a[0] !== b[0]) {
    return a[0] - b[0];
  }
  if (a[1] !== b[1]) {
    return a[1] - b[1];
  }
  if (a[2] !== b[2]) {
    return a[2] - b[2];
  }
  if (a[3] !== b[3]) {
    return a[3] - b[3];
  }
  return 0;
};

const getImageIndex = (imageNames: string[], imageName: string, expectedIndex: number) => {
  if (imageNames[expectedIndex] === imageName) {
    return expectedIndex;
  }
  return imageNames.indexOf(imageName);
};

export const getMasonryKeyboardNavigationTarget = ({
  columnCount,
  currentImageName,
  currentIndex,
  direction,
  imageNames,
  mountedItems,
}: GetMasonryKeyboardNavigationTargetArg): MasonryKeyboardNavigationTarget | null => {
  const fallbackTarget = getMasonryKeyboardFallbackTarget({
    columnCount,
    currentIndex,
    direction,
    imageNames,
  });
  const currentItem = currentImageName ? mountedItems.find((item) => item.imageName === currentImageName) : undefined;

  if (!currentItem) {
    return fallbackTarget;
  }

  let bestItem: MasonryKeyboardNavigationItem | null = null;
  let bestScore: MasonryKeyboardCandidateScore | null = null;

  for (const candidate of mountedItems) {
    if (
      candidate.imageName === currentItem.imageName ||
      !isMasonryKeyboardCandidateInDirection(currentItem, candidate, direction)
    ) {
      continue;
    }

    const score = getMasonryKeyboardCandidateScore(currentItem, candidate, direction);
    if (!bestScore || compareMasonryKeyboardCandidateScores(score, bestScore) < 0) {
      bestItem = candidate;
      bestScore = score;
    }
  }

  if (!bestItem) {
    return fallbackTarget;
  }

  const index = getImageIndex(imageNames, bestItem.imageName, bestItem.index);
  if (index === -1) {
    return fallbackTarget;
  }

  return { imageName: bestItem.imageName, index };
};

export const getMountedMasonryKeyboardNavigationItems = (rootEl: HTMLDivElement): MasonryKeyboardNavigationItem[] => {
  const items: MasonryKeyboardNavigationItem[] = [];

  for (const el of rootEl.querySelectorAll<HTMLElement>('[data-absolute-index]')) {
    const index = Number.parseInt(el.dataset.absoluteIndex ?? '', 10);
    const imageName = el.querySelector<HTMLElement>('[data-item-id]')?.dataset.itemId;
    if (!Number.isFinite(index) || !imageName) {
      continue;
    }

    const rect = el.getBoundingClientRect();
    items.push({
      imageName,
      index,
      rect: {
        bottom: rect.bottom,
        height: rect.height,
        left: rect.left,
        right: rect.right,
        top: rect.top,
        width: rect.width,
      },
    });
  }

  return items;
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

export const getMasonryScroller = (rootEl: HTMLDivElement): HTMLElement | null => {
  return rootEl.querySelector<HTMLElement>('[data-masonry-scroller], [data-testid="virtuoso-scroller"]');
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

export const isMasonryImageVisible = (rootEl: HTMLDivElement, imageName: string): boolean => {
  const scroller = getMasonryScroller(rootEl);
  const item = getMountedMasonryItem(rootEl, imageName);
  if (!scroller || !item) {
    return false;
  }

  return isMasonryItemRectVisible({
    itemRect: item.getBoundingClientRect(),
    scrollerRect: scroller.getBoundingClientRect(),
  });
};

const scrollScrollerByViewport = (scroller: HTMLElement, direction: Exclude<MasonryScrollDirection, null>) => {
  const amount = Math.max(1, scroller.clientHeight) * (direction === 'down' ? 1 : -1);

  if (typeof scroller.scrollBy === 'function') {
    scroller.scrollBy({ behavior: 'auto', top: amount });
  } else {
    scroller.scrollTop += amount;
  }
};

export const scrollMasonryImageIntoView = (arg: ScrollMasonryImageIntoViewArg): (() => void) => {
  let frame: number | null = null;
  let isCancelled = false;

  const scroll = (attempt: number) => {
    if (isCancelled) {
      return;
    }

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
    frame = requestAnimationFrame(() => {
      frame = null;
      scroll(attempt + 1);
    });
  };

  scroll(0);

  return () => {
    isCancelled = true;
    if (frame !== null) {
      cancelAnimationFrame(frame);
      frame = null;
    }
  };
};
