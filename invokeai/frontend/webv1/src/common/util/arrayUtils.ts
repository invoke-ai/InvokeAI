export function moveToStart<T>(array: T[], selectItemCallback: (item: T) => boolean): T[];
export function moveToStart<T>(array: T[], item: T): T[];
export function moveToStart<T>(array: T[], arg1: T | ((item: T) => boolean)): T[] {
  const index = arg1 instanceof Function ? array.findIndex(arg1) : array.indexOf(arg1);
  if (index > 0) {
    const [item] = array.splice(index, 1);
    //@ts-expect-error - These indicies are safe per the previous check
    array.unshift(item);
  }
  return array;
}

export function moveOneToStart<T>(array: T[], selectItemCallback: (item: T) => boolean): T[];
export function moveOneToStart<T>(array: T[], item: T): T[];
export function moveOneToStart<T>(array: T[], arg1: T | ((item: T) => boolean)): T[] {
  const index = arg1 instanceof Function ? array.findIndex(arg1) : array.indexOf(arg1);
  if (index > 0) {
    //@ts-expect-error - These indicies are safe per the previous check
    [array[index], array[index - 1]] = [array[index - 1], array[index]];
  }
  return array;
}

export function moveToEnd<T>(array: T[], selectItemCallback: (item: T) => boolean): T[];
export function moveToEnd<T>(array: T[], item: T): T[];
export function moveToEnd<T>(array: T[], arg1: T | ((item: T) => boolean)): T[] {
  const index = arg1 instanceof Function ? array.findIndex(arg1) : array.indexOf(arg1);
  if (index >= 0 && index < array.length - 1) {
    const [item] = array.splice(index, 1);
    //@ts-expect-error - These indicies are safe per the previous check
    array.push(item);
  }
  return array;
}

export function moveOneToEnd<T>(array: T[], selectItemCallback: (item: T) => boolean): T[];
export function moveOneToEnd<T>(array: T[], item: T): T[];
export function moveOneToEnd<T>(array: T[], arg1: T | ((item: T) => boolean)): T[] {
  const index = arg1 instanceof Function ? array.findIndex(arg1) : array.indexOf(arg1);
  if (index >= 0 && index < array.length - 1) {
    //@ts-expect-error - These indicies are safe per the previous check
    [array[index], array[index + 1]] = [array[index + 1], array[index]];
  }
  return array;
}
