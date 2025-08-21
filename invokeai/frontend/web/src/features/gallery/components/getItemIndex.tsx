/**
 * Get the index of the item in the list of item names.
 * If the item name is not found, return 0.
 * If no item name is provided, return 0.
 */
export const getItemIndex = (targetItemId: string | undefined | null, itemIds: string[]) => {
  if (!targetItemId || itemIds.length === 0) {
    return 0;
  }
  const index = itemIds.findIndex((n) => n === targetItemId);
  return index >= 0 ? index : 0;
};
