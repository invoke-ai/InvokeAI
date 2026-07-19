import type { ComponentType, ReactNode } from 'react';

export type ProviderComponent = ComponentType<{ children: ReactNode }>;

/**
 * Flattens a list of context providers into a single component. The array is
 * nesting order, outermost first: `composeProviders([A, B])` renders
 * `<A><B>{children}</B></A>`.
 *
 * MUST be called at module scope only. Each call creates a new component
 * identity, so composing inside render would remount the entire subtree on
 * every render.
 */
export const composeProviders = (providers: ReadonlyArray<ProviderComponent>): ProviderComponent => {
  const Composed = ({ children }: { children: ReactNode }): ReactNode =>
    providers.reduceRight<ReactNode>((wrapped, Provider) => <Provider>{wrapped}</Provider>, children);
  return Composed;
};
