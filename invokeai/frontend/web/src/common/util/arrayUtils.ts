export const moveForward = <T>(array: T[], callback: (item: T) => boolean): T[] => {
  const index = array.findIndex(callback);
  if (index >= 0 && index < array.length - 1) {
    //@ts-expect-error - These indicies are safe per the previous check
    [array[index], array[index + 1]] = [array[index + 1], array[index]];
  }
  return array;
};

export const moveToFront = <T>(array: T[], callback: (item: T) => boolean): T[] => {
  const index = array.findIndex(callback);
  if (index > 0) {
    const [item] = array.splice(index, 1);
    //@ts-expect-error - These indicies are safe per the previous check
    array.unshift(item);
  }
  return array;
};

export const moveBackward = <T>(array: T[], callback: (item: T) => boolean): T[] => {
  const index = array.findIndex(callback);
  if (index > 0) {
    //@ts-expect-error - These indicies are safe per the previous check
    [array[index], array[index - 1]] = [array[index - 1], array[index]];
  }
  return array;
};

export const moveToBack = <T>(array: T[], callback: (item: T) => boolean): T[] => {
  const index = array.findIndex(callback);
  if (index >= 0 && index < array.length - 1) {
    const [item] = array.splice(index, 1);
    //@ts-expect-error - These indicies are safe per the previous check
    array.push(item);
  }
  return array;
};
