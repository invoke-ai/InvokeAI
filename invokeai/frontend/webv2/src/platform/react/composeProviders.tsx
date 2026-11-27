import type { ComponentType, ReactNode } from 'react';

export type ProviderComponent = ComponentType<{ children: ReactNode }>;

/**
 * Outermost provider first. Call only at module scope: each call creates a component identity and render-time
 * composition remounts children.
 */
export const composeProviders = (providers: ReadonlyArray<ProviderComponent>): ProviderComponent => {
  const Composed = ({ children }: { children: ReactNode }): ReactNode =>
    providers.reduceRight<ReactNode>((wrapped, Provider) => <Provider>{wrapped}</Provider>, children);
  return Composed;
};
