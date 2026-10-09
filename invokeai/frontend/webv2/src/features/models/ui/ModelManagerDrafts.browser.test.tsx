import type { ModelConfig } from '@features/models/core/types';
import type * as ModelsApi from '@features/models/data/api';

import { ChakraProvider } from '@chakra-ui/react';
import { refreshModels, removeModelsFromStore, setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { MAX_MODEL_DRAFTS, recordModelDraftFields } from '@features/models/ui/modelDraftsStore';
import { closeModelDetail, openModelDetail, pruneModelsUiKeys } from '@features/models/ui/uiStore';
import { auditAccessibility } from '@platform/browser/auditAccessibility.testing';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { applyThemeToRoot } from '@theme/applyTheme';
import { DEFAULT_THEME_ID, system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { ModelManagerView } from './ModelManagerView';

/**
 * Unsaved model edits survive moving around the manager: switching tabs and models unmounts the detail forms, and
 * returning restores the draft, the open editor, and a failed save's error until the user saves or cancels.
 */

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
  // Panels the navigation passes through; they stay loading so the test touches no network.
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
// Inside the row's button, so the marker is part of the row's accessible name.
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

/** Leave the model's details through both other tabs and come back. */
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

/** A catalog refresh that finds the given server records, as after another client's change. */
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

  it('keeps a separate draft per model across model switches', async () => {
    await startEditing('Draft A');
    await selectModel(B.key);
    expect(isEditorOpen()).toBe(false);
    await startEditing('Draft B');

    await selectModel(A.key);
    expect(nameInput()?.value).toBe('Draft A');
    await selectModel(B.key);
    expect(nameInput()?.value).toBe('Draft B');
  });

  it('marks models with unsaved edits in the library and clears the mark when the edit is undone', async () => {
    expect(rowMarker(A.key)).toBeNull();

    await startEditing('Draft A');
    expect(rowMarker(A.key)).not.toBeNull();
    expect(rowMarker(B.key)).toBeNull();

    // An open editor with nothing changed is not an unsaved change.
    await fill(nameInput(), A.name);
    expect(isEditorOpen()).toBe(true);
    expect(rowMarker(A.key)).toBeNull();
    expect(textShown('models.unsavedChanges')).toBe(false);
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

  it('clears the draft after a successful save', async () => {
    api.updateModel.mockResolvedValue({ ...A, name: 'Saved A' });
    await startEditing('Saved A');

    await click(buttonNamed('users.saveChanges'));

    await expect.poll(isEditorOpen).toBe(false);
    expect(api.updateModel).toHaveBeenCalledWith(
      A.key,
      expect.objectContaining({ name: 'Saved A' }),
      expect.anything()
    );
    expect(rowMarker(A.key)).toBeNull();
    await visitOtherTabsAndReturn();
    expect(isEditorOpen()).toBe(false);
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

  it("drops a model's draft when a refresh finds the model gone", async () => {
    await startEditing('Draft A');
    await selectModel(B.key);

    await serverNowHas(B);
    // Reinstalled under the same key: it must start clean, not resurrect the old draft.
    await serverNowHas(A, B);

    expect(rowMarker(A.key)).toBeNull();
    await selectModel(A.key);
    expect(isEditorOpen()).toBe(false);
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

  it('takes server changes to untouched fields while keeping the edits, open or not', async () => {
    await startEditing('Draft A');

    await serverNowHas({ ...A, description: 'Written elsewhere' }, B);
    expect(nameInput()?.value).toBe('Draft A');
    expect(descriptionInput()?.value).toBe('Written elsewhere');

    await openTab('add');
    await serverNowHas({ ...A, description: 'Written again', source_url: 'https://example.com/a' }, B);
    await openTab('details');

    expect(nameInput()?.value).toBe('Draft A');
    expect(descriptionInput()?.value).toBe('Written again');
    expect(fieldControl<HTMLInputElement>('models.sourceUrl')?.value).toBe('https://example.com/a');
    expect(statusShown('models.changedElsewhere')).toBe(false);
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

  it('retains default-settings edits across navigation until Reset discards them', async () => {
    const weightSwitch = () =>
      fieldSwitch('models.customizeDefaultField:models.defaultFields.weight') as HTMLInputElement;

    await click(weightSwitch().closest('label')!);
    expect(weightSwitch().checked).toBe(true);

    await visitOtherTabsAndReturn();
    expect(weightSwitch().checked).toBe(true);
    expect(textShown('models.unsavedChanges')).toBe(true);
    expect(rowMarker(A.key)).not.toBeNull();

    await click(buttonNamed('common.reset'));
    expect(weightSwitch().checked).toBe(false);
    expect(rowMarker(A.key)).toBeNull();
  });

  it('settles an identity save that lands after leaving and returning, keeping edits typed meanwhile', async () => {
    const save = deferredSave();
    await startEditing('Sent name');
    await click(saveChangesButton());

    await visitOtherTabsAndReturn();
    // The remounted form still shows the request it did not start, and cannot send a second one.
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

  it('shows an identity save failure that lands after leaving and returning', async () => {
    const save = deferredSave();
    await startEditing('Taken name');
    await click(saveChangesButton());
    await visitOtherTabsAndReturn();

    await save.reject(new Error('A model with this name already exists.'));

    expect(document.querySelector('[role="alert"]')?.textContent).toBe('A model with this name already exists.');
    expect(nameInput()?.value).toBe('Taken name');
    expect(saveChangesButton().disabled).toBe(false);
  });

  it('settles a default-settings save that lands after leaving and returning', async () => {
    const save = deferredSave();
    await toggleDefault('weight');
    await click(saveDefaultsButton());
    await visitOtherTabsAndReturn();
    expect(saveDefaultsButton().disabled).toBe(true);
    await toggleDefault('weightMin');

    await save.resolve({ ...A, default_settings: { weight: 0.75 } });

    expect(api.updateModel).toHaveBeenCalledOnce();
    expect(defaultSwitch('weight').checked).toBe(true);
    expect(defaultSwitch('weightMin').checked).toBe(true);
    expect(textShown('models.unsavedChanges')).toBe(true);
    await click(buttonNamed('common.reset'));
    // Reset takes the saved record: the saved weight stays, the later edit goes.
    expect(defaultSwitch('weight').checked).toBe(true);
    expect(defaultSwitch('weightMin').checked).toBe(false);
  });

  it('keeps default-settings edits when their save fails after leaving and returning', async () => {
    const save = deferredSave();
    await toggleDefault('weight');
    await click(saveDefaultsButton());
    await visitOtherTabsAndReturn();

    await save.reject(new Error('Rejected.'));

    expect(defaultSwitch('weight').checked).toBe(true);
    expect(saveDefaultsButton().disabled).toBe(false);
    expect(rowMarker(A.key)).not.toBeNull();
  });

  it('stops calling an edit unsaved once the server holds the same value', async () => {
    await startEditing('Draft A');

    await serverNowHas({ ...A, name: 'Draft A' }, B);

    expect(isEditorOpen()).toBe(true);
    expect(rowMarker(A.key)).toBeNull();
    expect(textShown('models.unsavedChanges')).toBe(false);
  });

  it('rebases default settings onto server changes and flags an edited one the server also changed', async () => {
    await toggleDefault('weight');

    await serverNowHas({ ...A, default_settings: { weight: 0.5, weight_max: 1.5 } }, B);

    const weightInput = document.querySelectorAll<HTMLInputElement>('input[data-scope="number-input"]')[0];
    expect(weightInput?.value).toBe('0.75');
    expect(defaultSwitch('weightMax').checked).toBe(true);
    expect(statusShown('models.changedElsewhere')).toBe(true);

    await click(buttonNamed('common.reset'));
    expect(document.querySelectorAll<HTMLInputElement>('input[data-scope="number-input"]')[0]?.value).toBe('0.5');
    expect(statusShown('models.changedElsewhere')).toBe(false);
  });

  it('saves one surface while the other keeps its unsaved edits', async () => {
    api.updateModel.mockResolvedValue({ ...A, default_settings: { weight: 0.75 } });
    await startEditing('Draft A');
    await toggleDefault('weight');

    await click(saveDefaultsButton());

    await expect.poll(() => buttonNamed('common.reset')).toBeUndefined();
    expect(isEditorOpen()).toBe(true);
    expect(nameInput()?.value).toBe('Draft A');
    expect(rowMarker(A.key)).not.toBeNull();
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

  /** Below the 50rem both panes need: the library and the detail take turns. */
  const NARROW_PX = 640;
  const back = () => page.getByRole('button', { name: 'models.backToList' });
  const library = () => page.getByRole('heading', { name: 'models.title' });
  const detail = () => page.getByRole('tablist');
  const row = (key: string) => document.querySelector<HTMLElement>(`[data-list-row="${key}"] [data-list-primary]`)!;
  const setWidth = async (width: number) => {
    await act(async () => {
      host.style.width = `${String(width)}px`;
      await new Promise((resolve) => {
        requestAnimationFrame(resolve);
      });
    });
  };

  /** True when the control is inside the pane's visible scroll viewport, not just inside the manager's width. */
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
    // The install queue reports under the library as well as under the detail.
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

  it('reaches Add Models from the library header', async () => {
    await page.getByRole('button', { name: 'models.addModels', exact: true }).click();

    await expect.element(page.getByRole('tab', { name: 'models.addModels' })).toHaveAttribute('aria-selected', 'true');
    await expect.element(back()).toHaveFocus();
  });

  it('keeps an unsaved draft through Back, reopening, and wide and narrow resizes', async () => {
    await click(row(A.key));
    await startEditing('Draft A');

    await back().click();
    expect(rowMarker(A.key)).not.toBeNull();
    await click(row(A.key));
    expect(nameInput()?.value).toBe('Draft A');

    await setWidth(1200);
    await expect.element(library()).toBeVisible();
    await expect.element(back()).not.toBeInTheDocument();
    expect(nameInput()?.value).toBe('Draft A');

    await setWidth(NARROW_PX);
    await expect.element(back()).toBeVisible();
    expect(isEditorOpen()).toBe(true);
    expect(nameInput()?.value).toBe('Draft A');
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

  it('shows the detail when a model is opened from elsewhere, such as the install queue', async () => {
    await act(() => openModelDetail(A.key));

    await expect.element(detail()).toBeVisible();
    await expect.element(back()).toBeVisible();
  });
});

describe('model manager starting pane', () => {
  let host: HTMLDivElement;
  let root: Root;

  const library = () => page.getByRole('heading', { name: 'models.title' });
  const addTab = () => page.getByRole('tab', { name: 'models.addModels' });

  const mount = async (models: ModelConfig[]) => {
    // An account change resets the manager, so the starting pane is still undecided.
    accountLifecycle.activate('model-manager-start-test', ':user:model-manager-start-test');
    setModelsSnapshotForTests({ models, status: 'loaded' });
    host = document.createElement('div');
    host.style.cssText = 'display:flex;height:450px;width:640px;';
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

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('opens a non-empty library on the list', async () => {
    await mount([A, B]);

    await expect.element(library()).toBeVisible();
    await expect.element(addTab()).not.toBeInTheDocument();
  });

  it('opens an empty library on Add Models, and an arriving install leaves the pane, tab and focus alone', async () => {
    await mount([]);

    await expect.element(addTab()).toHaveAttribute('aria-selected', 'true');
    await expect.element(library()).not.toBeInTheDocument();
    // Settling the starting pane does not pull focus into an untouched page.
    expect(document.activeElement).toBe(document.body);
    const field = await vi.waitFor(() => {
      const input = document.querySelector<HTMLInputElement>(
        '[role="tabpanel"] input[type="text"], [role="tabpanel"] input:not([type])'
      );
      expect(input).not.toBeNull();
      return input!;
    });
    await act(() => userEvent.click(field));

    await act(() => setModelsSnapshotForTests({ models: [A], status: 'loaded' }));

    await expect.element(addTab()).toHaveAttribute('aria-selected', 'true');
    await expect.element(library()).not.toBeInTheDocument();
    expect(document.activeElement).toBe(field);
  });

  it('stays on the list, now empty, when the last model is deleted', async () => {
    await mount([A]);
    await expect.element(library()).toBeVisible();

    await act(() => {
      removeModelsFromStore([A.key]);
      pruneModelsUiKeys([A.key]);
    });

    await expect.element(library()).toBeVisible();
    await expect.element(addTab()).not.toBeInTheDocument();
  });
});

const defaultSwitch = (field: 'weight' | 'weightMin' | 'weightMax') =>
  fieldSwitch(`models.customizeDefaultField:models.defaultFields.${field}`)!;
const toggleDefault = (field: 'weight' | 'weightMin' | 'weightMax') =>
  click(defaultSwitch(field).closest('label') ?? undefined);
const buttonContaining = (text: string) =>
  [...document.querySelectorAll<HTMLButtonElement>('button')].find((button) => button.textContent.includes(text))!;
const saveChangesButton = () => buttonContaining('users.saveChanges');
const saveDefaultsButton = () => buttonContaining('models.saveDefaults');

/** A save the test settles by hand, after the user has moved around. */
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

const fieldSwitch = (label: string): HTMLInputElement | null => {
  const labelElement = [...document.querySelectorAll('[data-scope="switch"][data-part="label"]')].find(
    (element) => element.textContent === label
  );

  return labelElement?.closest('label')?.querySelector('input') ?? null;
};
