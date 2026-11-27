import type { ModelConfig, ModelIdentification } from '@features/models/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { type ModelsUiAdapter, ModelsUiProvider, NO_DEFAULTS_RULES } from '@features/models/ui/ModelsUiContext';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { ModelIdentificationNotice } from './ModelIdentificationNotice';

const api = vi.hoisted(() => ({
  getModelIdentification: vi.fn<(key: string) => Promise<ModelIdentification>>(),
}));

vi.mock('@features/models/data/api', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  getModelIdentification: api.getModelIdentification,
}));
vi.mock('@features/models/ui/useModelsNotify', () => ({
  useNotify: () => ({ error: vi.fn(), info: vi.fn(), success: vi.fn() }),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, string>) =>
      options ? `${key}(${Object.values(options).join(',')})` : key,
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const model = (key: string, fields: Partial<ModelConfig>) =>
  ({
    base: 'sd-1',
    file_size: 1024,
    format: 'checkpoint',
    hash: `hash-${key}`,
    key,
    name: key,
    path: `${key}.safetensors`,
    source: `${key}.safetensors`,
    source_type: 'path',
    type: 'main',
    variant: 'normal',
    ...fields,
  }) as unknown as ModelConfig;

const WRONG_BASE = model('wrong-base', {});
const UNKNOWN = model('unknown-1', { base: 'unknown', format: 'unknown', type: 'unknown', variant: null });
const CKPT = model('old-ckpt', { path: 'old-ckpt.ckpt' });
const PROPOSED_SDXL = { ...WRONG_BASE, base: 'sdxl' } as ModelConfig;

const adapter = (canManageModels: boolean): ModelsUiAdapter => ({
  canManageModels,
  enableModelDescriptions: true,
  installInPlace: true,
  isProjectActive: () => false,
  managerProjectId: null,
  defaultsRules: NO_DEFAULTS_RULES,
  setInstallInPlace: () => undefined,
});

const requestConfirm = vi.fn();

describe('ModelIdentificationNotice', () => {
  let host: HTMLDivElement;
  let root: Root;
  let queryClient: QueryClient;

  const render = async (candidate: ModelConfig, canManageModels = true) => {
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <QueryClientProvider client={queryClient}>
            <ModelsUiProvider adapter={adapter(canManageModels)}>
              <ModelIdentificationNotice model={candidate} onRequestConfirm={requestConfirm} />
            </ModelsUiProvider>
          </QueryClientProvider>
        </ChakraProvider>
      )
    );
  };
  const button = (label: string) => [...host.querySelectorAll('button')].find((el) => el.textContent === label);

  beforeEach(() => {
    queryClient = new QueryClient();
    api.getModelIdentification.mockReset();
    requestConfirm.mockReset();
    accountLifecycle.activate('identification-notice', ':user:identification-notice');
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('shows what the file looks like and asks for the re-identify confirmation on Apply', async () => {
    api.getModelIdentification.mockResolvedValue({ changed: true, proposed: PROPOSED_SDXL, refusal: null });
    await render(WRONG_BASE);

    await expect.poll(() => host.textContent).toContain('models.identificationLooksLike');
    // Only the field that differs is listed, from the stored label to the proposed one.
    expect(host.textContent).toContain('models.identificationChange(models.identificationFieldBase,SD 1.x,SDXL)');
    expect(host.textContent).not.toContain('models.identificationFieldFormat');
    expect(requestConfirm).not.toHaveBeenCalled();

    await act(() => userEvent.click(button('common.apply')!));

    // The record is not rewritten here: the shared confirmation warns that custom settings are reset, then applies.
    expect(requestConfirm).toHaveBeenCalledExactlyOnceWith({ kind: 'reidentify', model: WRONG_BASE });
  });

  it('checks an Unknown record on opening and hides the notice when dismissed', async () => {
    api.getModelIdentification.mockResolvedValue({
      changed: true,
      proposed: { ...UNKNOWN, base: 'flux', type: 'main' } as ModelConfig,
      refusal: null,
    });
    await render(UNKNOWN);

    await expect.poll(() => host.textContent).toContain('models.identificationLooksLike');
    await act(() => userEvent.click(button('models.identificationDismiss')!));

    expect(host.textContent).toBe('');
  });

  it('says why a recognised file is unusable', async () => {
    api.getModelIdentification.mockResolvedValue({ changed: false, proposed: null, refusal: 'no projection' });
    await render(UNKNOWN);

    await expect.poll(() => host.textContent).toContain('models.identificationRefused(no projection)');
    expect(button('common.apply')).toBeUndefined();
  });

  it('leaves a file that reads as filed alone', async () => {
    api.getModelIdentification.mockResolvedValue({ changed: false, proposed: WRONG_BASE, refusal: null });
    await render(WRONG_BASE);

    await expect.poll(() => api.getModelIdentification.mock.calls.length).toBe(1);
    expect(host.textContent).toBe('');
  });

  it('does not read a .ckpt file until asked, and then reports the result', async () => {
    api.getModelIdentification.mockResolvedValue({ changed: true, proposed: PROPOSED_SDXL, refusal: null });
    await render(CKPT);

    expect(host.textContent).toContain('models.identificationCheckHint');
    expect(api.getModelIdentification).not.toHaveBeenCalled();

    await act(() => userEvent.click(button('models.identificationCheck')!));

    await expect.poll(() => host.textContent).toContain('models.identificationLooksLike');
    expect(api.getModelIdentification).toHaveBeenCalledExactlyOnceWith(CKPT.key, expect.anything());
  });

  it('confirms a requested check that finds nothing to change, and offers a retry when it fails', async () => {
    api.getModelIdentification
      .mockRejectedValueOnce(new Error('boom'))
      .mockResolvedValueOnce({ changed: false, proposed: CKPT, refusal: null });
    await render(CKPT);

    await act(() => userEvent.click(button('models.identificationCheck')!));
    await expect.poll(() => host.textContent).toContain('models.identificationFailed');
    expect(host.textContent).toContain('boom');

    await act(() => userEvent.click(button('common.retry')!));
    await expect.poll(() => host.textContent).toContain('models.identificationMatches');
  });

  it('does not read the file again when the record changes after a requested check', async () => {
    api.getModelIdentification.mockResolvedValue({ changed: false, proposed: CKPT, refusal: null });
    await render(CKPT);
    await act(() => userEvent.click(button('models.identificationCheck')!));
    await expect.poll(() => host.textContent).toContain('models.identificationMatches');

    await render({ ...CKPT, base: 'sdxl' } as ModelConfig);

    expect(api.getModelIdentification).toHaveBeenCalledTimes(1);
    expect(button('models.identificationCheck')).toBeDefined();
  });

  it('never asks the backend when the user cannot manage models', async () => {
    api.getModelIdentification.mockResolvedValue({ changed: true, proposed: PROPOSED_SDXL, refusal: null });
    await render(UNKNOWN, false);
    await render(CKPT, false);

    expect(host.textContent).toBe('');
    expect(api.getModelIdentification).not.toHaveBeenCalled();
  });
});
