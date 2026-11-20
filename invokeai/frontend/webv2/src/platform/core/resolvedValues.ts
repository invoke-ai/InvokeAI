declare const resolvedValuesBrand: unique symbol;

/** Widget values with the user's defaults applied, so readers of layered keys cannot be handed stored pins. */
export type ResolvedValues = Readonly<Record<string, unknown>> & { readonly [resolvedValuesBrand]: true };
