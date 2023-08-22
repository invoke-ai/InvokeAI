/**
 * Comparator function for sorting dates in ascending order
 */
export const dateComparator = (a: string, b: string) => {
  const dateA = new Date(a);
  const dateB = new Date(b);

  // sort in ascending order
  if (dateA > dateB) {
    return 1;
  }
  if (dateA < dateB) {
    return -1;
  }
  return 0;
};
