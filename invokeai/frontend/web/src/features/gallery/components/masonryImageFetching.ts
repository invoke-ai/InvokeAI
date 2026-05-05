type GetMasonryPrefetchImageNamesArg = {
  cachedImageNames: string[];
  columnCount: number;
  imageNames: string[];
  mountedRange: { endIndex: number; startIndex: number } | null;
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

export const getUncachedMasonryImageNames = (imageNames: string[], cachedImageNames: string[]): string[] => {
  const cachedImageNamesSet = new Set(cachedImageNames);
  const uncachedImageNames: string[] = [];
  const seenImageNames = new Set<string>();

  for (const imageName of imageNames) {
    if (cachedImageNamesSet.has(imageName) || seenImageNames.has(imageName)) {
      continue;
    }
    seenImageNames.add(imageName);
    uncachedImageNames.push(imageName);
  }

  return uncachedImageNames;
};

export const getMasonryPrefetchImageNames = ({
  cachedImageNames,
  columnCount,
  imageNames,
  mountedRange,
}: GetMasonryPrefetchImageNamesArg): string[] => {
  if (!mountedRange || imageNames.length === 0) {
    return [];
  }

  const buffer = Math.max(columnCount * 32, 72);
  const startIndex = Math.max(0, mountedRange.startIndex - buffer);
  const endIndex = Math.min(imageNames.length - 1, mountedRange.endIndex + buffer);

  if (endIndex < startIndex) {
    return [];
  }

  return getUncachedMasonryImageNames(imageNames.slice(startIndex, endIndex + 1), cachedImageNames);
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
