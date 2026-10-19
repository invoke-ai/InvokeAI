const identities = new WeakMap<object, number>();
let nextIdentity = 1;

/** A short, process-local identity for immutable snapshots used in query keys. */
export const getObjectIdentity = (value: object | null | undefined, prefix = 'ref'): string => {
  if (value === null || value === undefined) {
    return `${prefix}:none`;
  }

  let identity = identities.get(value);

  if (identity === undefined) {
    identity = nextIdentity;
    nextIdentity += 1;
    identities.set(value, identity);
  }

  return `${prefix}:${identity}`;
};
