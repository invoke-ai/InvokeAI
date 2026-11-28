/**
 * A digest of the layout fields a document stored before the split, as saved. What this build saves beside them
 * (`heldView`) names the digest it was saved with, so it is read only beside that exact layout: a build before the split
 * that rearranges and saves the project drops the note, but one that copies it (a duplicate) keeps it beside another.
 */
export const digestStoredLayout = (fields: {
  floatingWidgets?: unknown;
  layout?: unknown;
  presetWorkingLayouts?: unknown;
  widgetRegions?: unknown;
}): string => {
  const text = JSON.stringify([
    fields.floatingWidgets ?? null,
    fields.layout ?? null,
    fields.presetWorkingLayouts ?? null,
    fields.widgetRegions ?? null,
  ]);
  // FNV-1a: telling one saved layout from another needs no more.
  let hash = 0x811c9dc5;

  for (let index = 0; index < text.length; index += 1) {
    hash = Math.imul(hash ^ text.charCodeAt(index), 0x01000193);
  }

  return (hash >>> 0).toString(16);
};
