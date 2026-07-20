export type ProductionPortBindingOwner =
  | 'gallery'
  | 'generation'
  | 'models'
  | 'platform'
  | 'queue'
  | 'upscale'
  | 'workflow';

export type ProductionPortBindingLifetime = 'pre-react' | 'authenticated-session' | 'editor' | 'lazy-editor';

export interface ProductionPortBinding {
  id: string;
  lifetime: ProductionPortBindingLifetime;
  owner: ProductionPortBindingOwner;
}

/** Complete, implementation-free inventory of App's production bindings. */
export const productionPortBindingManifest = [
  { id: 'platform.http-auth', lifetime: 'pre-react', owner: 'platform' },
  { id: 'platform.authenticated-socket', lifetime: 'authenticated-session', owner: 'platform' },
  { id: 'queue.workbench-runtime', lifetime: 'editor', owner: 'queue' },
  { id: 'gallery.image-actions-bridge', lifetime: 'lazy-editor', owner: 'gallery' },
  { id: 'models.ui', lifetime: 'editor', owner: 'models' },
  { id: 'queue.ui', lifetime: 'editor', owner: 'queue' },
  { id: 'gallery.ui', lifetime: 'editor', owner: 'gallery' },
  { id: 'generation.ui', lifetime: 'editor', owner: 'generation' },
  { id: 'upscale.ui', lifetime: 'editor', owner: 'upscale' },
  { id: 'workflow.ui', lifetime: 'editor', owner: 'workflow' },
] as const satisfies readonly ProductionPortBinding[];

export type ProductionPortBindingId = (typeof productionPortBindingManifest)[number]['id'];

const bindingIds = new Set<ProductionPortBindingId>(productionPortBindingManifest.map((binding) => binding.id));

/** Typed mount declaration used by lifecycle-specific composition modules. */
export const bindProductionPortBinding = <Mount>(id: ProductionPortBindingId, mount: Mount): Mount => {
  if (!bindingIds.has(id)) {
    throw new Error(`Unknown production port binding: ${id}`);
  }
  return mount;
};

/** Typed one-shot mount for the pre-React binding. */
export const mountProductionPortBinding = <Result>(id: ProductionPortBindingId, mount: () => Result): Result => {
  bindProductionPortBinding(id, mount);
  return mount();
};
