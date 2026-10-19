import { z } from 'zod';

// Schemas and types for working with semver

const zVersionInt = z.coerce.number().int().min(0);

export const zSemVer = z.string().refine((val) => {
  const [major, minor, patch] = val.split('.');
  return (
    zVersionInt.safeParse(major).success && zVersionInt.safeParse(minor).success && zVersionInt.safeParse(patch).success
  );
});
