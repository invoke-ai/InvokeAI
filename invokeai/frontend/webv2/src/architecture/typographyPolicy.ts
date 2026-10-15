import { collectTypographyLiterals } from './tsSourceAnalysis';

/** Chakra accepts arbitrary CSS strings, so its generated types cannot reject the retired token. */
export const checkTypographySource = (sourcePath: string, source: string): string[] =>
  collectTypographyLiterals(sourcePath, source)
    .filter(({ value }) => value === '2xs')
    .map(
      ({ column, line, property }) =>
        `${sourcePath}:${line}:${column}: ${property} uses retired typography token "2xs"; use "xs" for 10px type.`
    );
