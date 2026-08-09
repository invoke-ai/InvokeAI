/** Fallback display casing for open-union taxonomy values ("t5_encoder" -> "T5 Encoder"). */
export const toTitleCase = (value: string): string =>
  value.replaceAll(/[_-]+/g, ' ').replace(/\w\S*/g, (word) => word.charAt(0).toUpperCase() + word.slice(1));
