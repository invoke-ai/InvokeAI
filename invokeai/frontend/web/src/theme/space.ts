const getSpaceValues = (fractionOfDefault = 0.75) => {
  const spaceKeys = [
    0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 28,
    32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 96,
  ];

  const spaceObject = spaceKeys.reduce(
    (acc, val) => {
      acc[val] = `${val * (0.25 * fractionOfDefault)}rem`;
      return acc;
    },
    { px: '1px' } as Record<string, string>
  );

  return spaceObject;
};

export const space = getSpaceValues(0.75);
