/**
 * Its own file so the architecture table's store starts never requested: once anything asks for the table, the store
 * asks again on every account activation, which would hide a request made at mount.
 */

import type { ModelConfig } from '@features/models';
import type { ArchitectureCapabilitiesRow } from '@features/models/architectures';

import { ChakraProvider } from '@chakra-ui/react';
import { getArchitectureCapabilitiesSnapshot } from '@features/models/data/architectureCapabilitiesStore';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { ModelsUiProvider, NO_DEFAULTS_RULES } from '@features/models/ui/ModelsUiContext';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { expect, it, vi } from 'vitest';

import { RelatedModelsSection } from './RelatedModelsSection';

vi.mock('@features/models/data/relationshipsApi', () => ({
  addModelRelationship: vi.fn(),
  getRelatedModelKeys: vi.fn().mockResolvedValue([]),
  removeModelRelationship: vi.fn(),
}));
const getArchitectureCapabilities = vi.hoisted(() =>
  vi.fn<() => Promise<ArchitectureCapabilitiesRow[]>>(() => new Promise(() => {}))
);
vi.mock('@features/models/data/architectureCapabilitiesApi', () => ({
  getArchitectureCapabilities: () => getArchitectureCapabilities(),
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const model = (overrides: Partial<ModelConfig>): ModelConfig =>
  ({
    description: null,
    file_size: 1024,
    format: 'checkpoint',
    hash: 'hash',
    source: 'model',
    ...overrides,
  }) as ModelConfig;

const MODELS_UI_ADAPTER = {
  canManageModels: true,
  enableModelDescriptions: false,
  installInPlace: true,
  isProjectActive: () => true,
  managerProjectId: null,
  defaultsRules: NO_DEFAULTS_RULES,
  setInstallInPlace: () => undefined,
};
const SECTION_MODEL = { base: 'flux', key: 'flux-main', type: 'main' } as const;

it('requests the architecture table when its picker opens, not when the section mounts', async () => {
  accountLifecycle.activate('related-models-demand', ':user:related-models-demand');
  setModelsSnapshotForTests({
    error: null,
    models: [
      model({ base: 'flux', key: 'flux-main', name: 'FLUX Dev', type: 'main' }),
      model({ base: 'flux', key: 'flux-vae', name: 'FLUX VAE', type: 'vae' }),
    ],
    status: 'loaded',
  });
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);

  await act(() => {
    root.render(
      <ChakraProvider value={system}>
        <ModelsUiProvider adapter={MODELS_UI_ADAPTER}>
          <RelatedModelsSection model={SECTION_MODEL} onError={vi.fn()} />
        </ModelsUiProvider>
      </ChakraProvider>
    );
  });

  expect(getArchitectureCapabilities).not.toHaveBeenCalled();
  expect(getArchitectureCapabilitiesSnapshot().status).toBe('idle');

  await act(() => host.querySelector<HTMLButtonElement>('[aria-haspopup="listbox"]')?.click());

  expect(getArchitectureCapabilities).toHaveBeenCalledOnce();
  expect(getArchitectureCapabilitiesSnapshot().status).toBe('loading');

  await act(() => root.unmount());
  host.remove();
  document.querySelectorAll('[data-scope="popover"]').forEach((element) => element.remove());
  accountLifecycle.invalidate();
});
