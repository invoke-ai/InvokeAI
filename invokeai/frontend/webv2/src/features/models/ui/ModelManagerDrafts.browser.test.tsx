import type { ModelConfig } from '@features/models/core/types';
import type * as ModelsApi from '@features/models/data/api';

import { ChakraProvider } from '@chakra-ui/react';
import { refreshModels, setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { MAX_MODEL_DRAFTS, recordModelDraftFields } from '@features/models/ui/modelDraftsStore';
import { closeModelDetail, openModelDetail } from '@features/models/ui/uiStore';
import { auditAccessibility } from '@platform/browser/auditAccessibility.testing';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { applyThemeToRoot } from '@theme/applyTheme';
import { DEFAULT_THEME_ID, system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { ModelManagerView } from './ModelManagerView';

const { api, pending } = vi.hoisted(() => ({
  api: {
    getModelsDir: vi.fn(() => Promise.resolve('/models')),
    listMissingModels: vi.fn(() => Promise.resolve([])),
    deleteModel: vi.fn(),
    listModels: vi.fn(),
    updateModel: vi.fn(),
  },
  pending: () => new Promise<never>(() => {}),
}));

vi.mock('@features/models/data/api', async (importOriginal) => ({
  ...(await importOriginal<typeof ModelsApi>()),
  ...api,
  getExternalProviderConfigs: pending,
  getFp8StorageSupport: pending,
  getHFTokenStatus: pending,
  getStarterModels: pending,
  listModelInstalls: pending,
}));
vi.mock('@features/models/data/relationshipsApi', () => ({
  addModelRelationship: pending,
  getRelatedModelKeys: () => Promise.resolve([]),
  removeModelRelationship: pending,
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) =>
      values ? `${key}:${Object.values(values).map(String).join(',')}` : key,
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const lora = (key: string, overrides: Partial<ModelConfig> = {}): ModelConfig =>
  ({
    base: 'sdxl',
    default_settings: null,
    description: null,
    file_size: 1024,
    format: 'lycoris',
    hash: `hash-${key}`,
    key,
    name: `Lora ${key}`,
    path: `/models/${key}.safetensors`,
    source: `/models/${key}.safetensors`,
    source_type: 'path',
    trigger_phrases: [],
    type: 'lora',
    ...overrides,
  }) as ModelConfig;

const A = lora('a');
const B = lora('b');

const buttonNamed = (name: string): HTMLButtonElement | undefined =>
  [...document.querySelectorAll<HTMLButtonElement>('button')].find((button) => button.textContent === name);

const click = async (element: HTMLElement | undefined) => {
  expect(element).toBeDefined();
  await act(() => userEvent.click(element!));
};

const fieldControl = <Control extends HTMLElement>(label: string): Control | null => {
  const labelElement = [...document.querySelectorAll('label')].find((element) => element.textContent === label);

  return labelElement ? (document.getElementById(labelElement.htmlFor) as Control | null) : null;
};

const nameInput = () => fieldControl<HTMLInputElement>('common.name');
const descriptionInput = () => fieldControl<HTMLTextAreaElement>('models.description');

const fill = async (control: HTMLInputElement | HTMLTextAreaElement | null, value: string) => {
  expect(control).not.toBeNull();
  await act(() => userEvent.fill(control!, value));
};

const isEditorOpen = () => buttonNamed('models.editing') !== undefined;
const textShown = (text: string) => document.body.textContent.includes(text);
const rowMarker = (key: string) =>
  document.querySelector(
    `[data-list-row="${key}"] [data-list-primary] [role="img"][aria-label="models.unsavedChanges"]`
  );
const statusShown = (text: string) =>
  [...document.querySelectorAll('[role="status"]')].some((element) => element.textContent === text);

const TAB_INDEX = { add: 1, details: 0, keys: 2 } as const;
const openTab = (tab: keyof typeof TAB_INDEX) =>
  click([...document.querySelectorAll<HTMLElement>('[role="tab"]')][TAB_INDEX[tab]]);
const selectModel = (key: string) =>
  click(document.querySelector<HTMLElement>(`[data-list-row="${key}"] [data-list-primary]`)!);

const visitOtherTabsAndReturn = async () => {
  await openTab('add');
  expect(nameInput()).toBeNull();
  await openTab('keys');
  await openTab('details');
};

const startEditing = async (name: string) => {
  await click(buttonNamed('common.edit'));
  await fill(nameInput(), name);
};

const serverNowHas = async (...models: ModelConfig[]) => {
  api.listModels.mockResolvedValueOnce(models);
  await act(() => refreshModels());
};

describe('model manager drafts', () => {
  let host: HTMLDivElement;
  let root: Root;

  const mount = async () => {
    host = document.createElement('div');
    host.style.height = '900px';
    host.style.width = '1200px';
    host.style.display = 'flex';
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <ModelManagerView />
        </ChakraProvider>
      )
    );
  };

  beforeEach(async () => {
    api.listModels.mockReset();
    api.updateModel.mockReset();
    accountLifecycle.activate('model-drafts-test', ':user:model-drafts-test');
    setModelsSnapshotForTests({ models: [A, B], status: 'loaded' });
    openModelDetail(A.key);
    await mount();
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('restores the open editor and its edits after visiting Add Models and API Keys', async () => {
    await startEditing('Draft A');
    await fill(descriptionInput(), 'Notes in progress');

    await visitOtherTabsAndReturn();

    expect(isEditorOpen()).toBe(true);
    expect(nameInput()?.value).toBe('Draft A');
    expect(descriptionInput()?.value).toBe('Notes in progress');
    expect(textShown('models.unsavedChanges')).toBe(true);
  });

  it('discards the draft on Cancel, and it stays discarded after navigating away and back', async () => {
    await startEditing('Draft A');

    await click(buttonNamed('common.cancel'));
    expect(isEditorOpen()).toBe(false);
    expect(rowMarker(A.key)).toBeNull();

    await visitOtherTabsAndReturn();
    expect(isEditorOpen()).toBe(false);
    await click(buttonNamed('common.edit'));
    expect(nameInput()?.value).toBe(A.name);
  });

  it('keeps the draft and the failure after a failed save while the user looks elsewhere', async () => {
    api.updateModel.mockRejectedValue(new Error('A model with this name already exists.'));
    await startEditing('Taken name');

    await click(buttonNamed('users.saveChanges'));
    await expect
      .poll(() => document.querySelector('[role="alert"]')?.textContent)
      .toBe('A model with this name already exists.');

    await visitOtherTabsAndReturn();

    expect(isEditorOpen()).toBe(true);
    expect(nameInput()?.value).toBe('Taken name');
    expect(document.querySelector('[role="alert"]')?.textContent).toBe('A model with this name already exists.');
    expect(rowMarker(A.key)).not.toBeNull();
  });

  it('forgets drafts on an account transition', async () => {
    await startEditing('Draft A');

    await act(() => {
      accountLifecycle.activate('model-drafts-test-other', ':user:model-drafts-test-other');
      setModelsSnapshotForTests({ models: [A, B], status: 'loaded' });
      openModelDetail(A.key);
    });

    expect(rowMarker(A.key)).toBeNull();
    expect(isEditorOpen()).toBe(false);
  });

  it('keeps an edited field the server also changed, says so, and Cancel takes the server value', async () => {
    await startEditing('Draft A');

    await serverNowHas({ ...A, name: 'Renamed elsewhere' }, B);

    expect(nameInput()?.value).toBe('Draft A');
    expect(statusShown('models.changedElsewhere')).toBe(true);

    await visitOtherTabsAndReturn();
    expect(statusShown('models.changedElsewhere')).toBe(true);

    await click(buttonNamed('common.cancel'));
    await click(buttonNamed('common.edit'));
    expect(nameInput()?.value).toBe('Renamed elsewhere');
    expect(statusShown('models.changedElsewhere')).toBe(false);
  });

  it('settles an identity save that lands after leaving and returning, keeping edits typed meanwhile', async () => {
    const save = deferredSave();
    await startEditing('Sent name');
    await click(saveChangesButton());

    await visitOtherTabsAndReturn();
    expect(saveChangesButton().disabled).toBe(true);
    await fill(descriptionInput(), 'Typed after saving');

    await save.resolve({ ...A, name: 'Sent name' });

    expect(api.updateModel).toHaveBeenCalledOnce();
    expect(saveChangesButton().disabled).toBe(false);
    expect(nameInput()?.value).toBe('Sent name');
    expect(descriptionInput()?.value).toBe('Typed after saving');
    expect(rowMarker(A.key)).not.toBeNull();
    expect(statusShown('models.changedElsewhere')).toBe(false);
  });

  it('keeps only the most recently edited drafts', async () => {
    const library = Array.from({ length: MAX_MODEL_DRAFTS + 1 }, (_, index) => lora(`m${index}`));

    await act(() => setModelsSnapshotForTests({ models: library, status: 'loaded' }));
    for (const model of library.slice(0, MAX_MODEL_DRAFTS)) {
      await act(() => recordModelDraftFields(model.key, 'identity', { name: 'Edited' }, { name: model.name }));
    }
    expect(rowMarker('m0')).not.toBeNull();

    await selectModel(`m${MAX_MODEL_DRAFTS}`);
    await startEditing('One draft too many');

    expect(rowMarker('m0')).toBeNull();
    expect(rowMarker('m1')).not.toBeNull();
  });
});

describe('model manager in a single pane', () => {
  let host: HTMLDivElement;
  let root: Root;

  const NARROW_PX = 640;
  const back = () => page.getByRole('button', { name: 'models.backToList' });
  const library = () => page.getByRole('heading', { name: 'models.title' });
  const detail = () => page.getByRole('tablist');
  const row = (key: string) => document.querySelector<HTMLElement>(`[data-list-row="${key}"] [data-list-primary]`)!;
  const isInView = (element: HTMLElement) => {
    const bounds = element.getBoundingClientRect();
    const viewport = element.closest('[data-scope="scroll-area"][data-part="viewport"]') ?? host;
    const visible = viewport.getBoundingClientRect();

    return (
      bounds.left >= visible.left &&
      bounds.right <= visible.right &&
      bounds.top >= visible.top &&
      bounds.bottom <= visible.bottom
    );
  };

  beforeEach(async () => {
    applyThemeToRoot(DEFAULT_THEME_ID);
    api.listModels.mockReset();
    api.deleteModel.mockReset();
    accountLifecycle.activate('model-manager-single-pane-test', ':user:model-manager-single-pane-test');
    setModelsSnapshotForTests({ models: [A, B], status: 'loaded' });
    closeModelDetail();
    host = document.createElement('div');
    host.style.cssText = `display:flex;height:450px;width:${String(NARROW_PX)}px;`;
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <ModelManagerView />
        </ChakraProvider>
      )
    );
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('opens a model from the library, keeps its actions in view, and goes back to the row', async () => {
    await expect.element(library()).toBeVisible();
    await expect.element(detail()).not.toBeInTheDocument();
    await expect.element(page.getByRole('button', { name: /models.installQueue/ })).toBeVisible();
    expect(await auditAccessibility(host)).toEqual([]);

    await click(row(B.key));

    await expect.element(back()).toHaveFocus();
    await expect.element(library()).not.toBeInTheDocument();
    const edit = page.getByRole('button', { name: 'common.edit', exact: true });
    const actions = page.getByRole('button', { name: 'models.actions', exact: true });
    for (const control of [edit, actions]) {
      await expect.element(control).toBeVisible();
      expect(isInView(control.element() as HTMLElement)).toBe(true);
    }
    await expect.element(page.getByRole('button', { name: /models.installQueue/ })).toBeVisible();
    expect(await auditAccessibility(host)).toEqual([]);
    await edit.click();
    expect(isEditorOpen()).toBe(true);

    await back().click();

    await expect.element(library()).toBeVisible();
    expect(document.activeElement).toBe(row(B.key));
    expect(row(B.key).getAttribute('aria-current')).toBe('true');
  });

  it('returns to the library with focus on a row after deleting the open model', async () => {
    api.deleteModel.mockResolvedValue(undefined);
    await click(row(B.key));

    await page.getByRole('button', { name: 'models.actions', exact: true }).click();
    await page.getByRole('menuitem', { name: 'models.deleteModel' }).click();
    await page.getByRole('alertdialog').getByRole('button', { name: 'models.deleteModel' }).click();

    await expect.element(page.getByRole('alertdialog')).not.toBeInTheDocument();
    await expect.element(library()).toBeVisible();
    await vi.waitFor(() => expect(document.activeElement).toBe(row(A.key)));
    expect(api.deleteModel).toHaveBeenCalledWith(B.key, expect.anything());
  });
});
const buttonContaining = (text: string) =>
  [...document.querySelectorAll<HTMLButtonElement>('button')].find((button) => button.textContent.includes(text))!;
const saveChangesButton = () => buttonContaining('users.saveChanges');
const deferredSave = () => {
  let resolve!: (model: ModelConfig) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<ModelConfig>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });

  api.updateModel.mockReturnValueOnce(promise);

  return {
    reject: (error: Error) =>
      act(async () => {
        reject(error);
        await promise.catch(() => undefined);
      }),
    resolve: (model: ModelConfig) =>
      act(async () => {
        resolve(model);
        await promise;
      }),
  };
};
