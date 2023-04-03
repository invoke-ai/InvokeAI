export const extractTimestampFromResultImageName = (imageName: string) => {
  const timestamp = imageName.split('_')?.pop()?.split('.')[0];

  if (timestamp === undefined) {
    return 0;
  }

  return Number(timestamp);
};
