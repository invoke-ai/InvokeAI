type GetMasonryPrefetchImageNamesArg = {
  cachedImageNames: string[];
  batchSize?: number;
  columnCount: number;
  imageNames: string[];
  inFlightImageNames?: Iterable<string>;
  mountedRange: { endIndex: number; startIndex: number } | null;
  scrollDirection?: MasonryScrollDirection;
};

type GetMasonryWarmupImageNamesArg = {
  batchSize: number;
  cachedImageNames: string[];
  imageNames: string[];
  inFlightImageNames?: Iterable<string>;
  maxImageCount: number;
  skippedImageNames?: Iterable<string>;
};

type GetShouldScheduleNextMasonryWarmupBatchArg = {
  didFetchBatch: boolean;
  imageNamesToFetchCount: number;
  isCancelled: boolean;
};

type GetMasonryInitialItemCountArg = {
  columnCount: number;
  imageCount: number;
  initialItemCountLimit: number;
  itemsPerColumn: number;
  minimumInitialItemCount: number;
};

type GetMasonryInFlightImageNamesArg = {
  backgroundInFlightImageNames: ReadonlyMap<string, number>;
  visibleInFlightImageNames: Iterable<string>;
};

type GetMasonrySkippedImageNamesArg = {
  requestedImageNames: string[];
  returnedImageNames: Iterable<string>;
};

type SetMasonryBackgroundInFlightImageNamesArg = {
  backgroundInFlightImageNames: Map<string, number>;
  imageNames: string[];
  requestId: number;
};

type StaticMasonryImageDimensions = {
  height: number;
  width: number;
};

type StaticMasonryColumnItem = {
  imageName: string;
  index: number;
};

type GetStaticMasonryColumnsArg = {
  columnCount: number;
  imageDimensionsByName: ReadonlyMap<string, StaticMasonryImageDimensions>;
  imageNames: string[];
};

type MasonryScrollDirection = 'down' | 'up' | null;

export const getUncachedMasonryImageNames = (
  imageNames: string[],
  cachedImageNames: string[],
  inFlightImageNames: Iterable<string> = []
): string[] => {
  const cachedImageNamesSet = new Set(cachedImageNames);
  const inFlightImageNamesSet = new Set(inFlightImageNames);
  const uncachedImageNames: string[] = [];
  const seenImageNames = new Set<string>();

  for (const imageName of imageNames) {
    if (cachedImageNamesSet.has(imageName) || inFlightImageNamesSet.has(imageName) || seenImageNames.has(imageName)) {
      continue;
    }
    seenImageNames.add(imageName);
    uncachedImageNames.push(imageName);
  }

  return uncachedImageNames;
};

export const getMasonryPrefetchImageNames = ({
  batchSize = Number.POSITIVE_INFINITY,
  cachedImageNames,
  columnCount,
  imageNames,
  inFlightImageNames,
  mountedRange,
  scrollDirection = 'down',
}: GetMasonryPrefetchImageNamesArg): string[] => {
  if (!mountedRange || imageNames.length === 0) {
    return [];
  }

  const buffer = Math.max(columnCount * 64, 144);
  const startIndex = Math.max(0, mountedRange.startIndex - buffer);
  const endIndex = Math.min(imageNames.length - 1, mountedRange.endIndex + buffer);

  if (endIndex < startIndex) {
    return [];
  }

  const mountedImageNames = imageNames.slice(mountedRange.startIndex, mountedRange.endIndex + 1);
  const previousImageNames = imageNames.slice(startIndex, mountedRange.startIndex).reverse();
  const nextImageNames = imageNames.slice(mountedRange.endIndex + 1, endIndex + 1);
  const prioritizedImageNames =
    scrollDirection === 'up'
      ? [...mountedImageNames, ...previousImageNames, ...nextImageNames]
      : [...mountedImageNames, ...nextImageNames, ...previousImageNames];

  return getUncachedMasonryImageNames(prioritizedImageNames, cachedImageNames, inFlightImageNames).slice(0, batchSize);
};

export const getMasonryWarmupImageNames = ({
  batchSize,
  cachedImageNames,
  imageNames,
  inFlightImageNames,
  maxImageCount,
  skippedImageNames = [],
}: GetMasonryWarmupImageNamesArg): string[] => {
  if (batchSize <= 0 || maxImageCount <= 0) {
    return [];
  }

  return getUncachedMasonryImageNames(
    imageNames.slice(0, maxImageCount),
    cachedImageNames,
    new Set([...(inFlightImageNames ?? []), ...skippedImageNames])
  ).slice(0, batchSize);
};

export const getShouldScheduleNextMasonryWarmupBatch = ({
  didFetchBatch,
  imageNamesToFetchCount,
  isCancelled,
}: GetShouldScheduleNextMasonryWarmupBatchArg): boolean => {
  return !isCancelled && didFetchBatch && imageNamesToFetchCount > 0;
};

export const getMasonryInitialItemCount = ({
  columnCount,
  imageCount,
  initialItemCountLimit,
  itemsPerColumn,
  minimumInitialItemCount,
}: GetMasonryInitialItemCountArg): number => {
  return Math.min(imageCount, initialItemCountLimit, Math.max(columnCount * itemsPerColumn, minimumInitialItemCount));
};

export const getMasonryInFlightImageNames = ({
  backgroundInFlightImageNames,
  visibleInFlightImageNames,
}: GetMasonryInFlightImageNamesArg): string[] => {
  return [...visibleInFlightImageNames, ...backgroundInFlightImageNames.keys()];
};

export const getMasonrySkippedImageNames = ({
  requestedImageNames,
  returnedImageNames,
}: GetMasonrySkippedImageNamesArg): string[] => {
  const returnedImageNamesSet = new Set(returnedImageNames);
  return requestedImageNames.filter((imageName) => !returnedImageNamesSet.has(imageName));
};

export const setMasonryBackgroundInFlightImageNames = ({
  backgroundInFlightImageNames,
  imageNames,
  requestId,
}: SetMasonryBackgroundInFlightImageNamesArg): void => {
  for (const imageName of imageNames) {
    backgroundInFlightImageNames.set(imageName, requestId);
  }
};

export const deleteMasonryBackgroundInFlightImageNames = ({
  backgroundInFlightImageNames,
  imageNames,
  requestId,
}: SetMasonryBackgroundInFlightImageNamesArg): void => {
  for (const imageName of imageNames) {
    if (backgroundInFlightImageNames.get(imageName) === requestId) {
      backgroundInFlightImageNames.delete(imageName);
    }
  }
};

export const getStaticMasonryColumns = ({
  columnCount,
  imageDimensionsByName,
  imageNames,
}: GetStaticMasonryColumnsArg): StaticMasonryColumnItem[][] => {
  const columns = Array.from({ length: columnCount }, () => [] as StaticMasonryColumnItem[]);
  const columnHeights = Array.from({ length: columnCount }, () => 0);

  imageNames.forEach((imageName, index) => {
    let shortestColumnIndex = 0;
    for (let columnIndex = 1; columnIndex < columnHeights.length; columnIndex++) {
      if (columnHeights[columnIndex]! < columnHeights[shortestColumnIndex]!) {
        shortestColumnIndex = columnIndex;
      }
    }
    columns[shortestColumnIndex]?.push({ imageName, index });

    const dimensions = imageDimensionsByName.get(imageName);
    const aspectHeight = dimensions && dimensions.width > 0 ? dimensions.height / dimensions.width : 1;
    columnHeights[shortestColumnIndex] = columnHeights[shortestColumnIndex]! + aspectHeight;
  });

  return columns;
};
