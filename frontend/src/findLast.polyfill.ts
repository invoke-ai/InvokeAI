/* eslint @typescript-eslint/ban-ts-comment: 0 */ 
/* eslint @typescript-eslint/no-explicit-any: 0 */ 
// @ts-ignore
Array.prototype.findLast = function <T, S extends T>(
  predicate: (value: T, index: number, obj: T[]) => value is S,
  thisArg?: any
) {
  let found: S | undefined = undefined;
  (thisArg ?? this).forEach((value: T, index: number, array: T[]) => {
    found = predicate(value, index, array) ? value : found;
  });
  return found;
};
