/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { GenerationModelCatalogItem } from '@features/generation/contracts';
import type { GenerationIssue, GenerationIssueFix } from '@features/generation/core/generationIssues';
import type { GenerateModelConfig, GenerateReferenceImage, GenerateSettings } from '@features/generation/core/types';
import type { GenerateSettingsUpdate } from '@features/generation/ui/generateDebounce';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings, getMaxReferenceImages } from '@features/generation/core/baseGenerationPolicies';
import { type GenerationUiAdapter, GenerationUiProvider } from '@features/generation/ui/GenerationUiContext';
import { revealGenerateIssue } from '@features/generation/ui/revealGenerateIssue';
import { GenerateIssuesProvider } from '@features/generation/ui/shared/GenerateIssues';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act, createRef, type Ref, useCallback, useImperativeHandle, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { commands, userEvent } from 'vitest/browser';

import { GenerateReferenceImagesContent } from './GenerateReferenceImagesSection';
import { describeReferenceImage, getPositionalDefaultNameTest } from './ReferenceImageRow';

declare module 'vitest/browser' {
  interface BrowserCommands {
    imeCompose: (steps: readonly { kind: 'commit' | 'compose'; text: string }[]) => Promise<void>;
  }
}

const i18n = i18next.createInstance();
// The shipped copy, so names and labels read as a user sees them.
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

seedArchitectureCapabilities();

const SDXL: GenerateModelConfig = { base: 'sdxl', format: 'diffusers', key: 'sdxl', name: 'SDXL', type: 'main' };
const FLUX: GenerateModelConfig = {
  base: 'flux',
  format: 'diffusers',
  key: 'flux-dev',
  name: 'FLUX dev',
  type: 'main',
};
const COGVIEW: GenerateModelConfig = { base: 'cogview4', key: 'cogview4', name: 'CogView4', type: 'main' };
const REDUX = { base: 'flux', key: 'redux', name: 'FLUX Redux', type: 'flux_redux' };
const NO_MODELS: [] = [];
const SDXL_ADAPTERS = [
  { base: 'sdxl', key: 'ip-adapter-sdxl', name: 'IP Adapter SDXL', type: 'ip_adapter' },
] as unknown as GenerationModelCatalogItem[];
const openManager = vi.fn();
const NO_VISION_COMPONENTS = {};
const undoable = vi.fn();
/** The models a model picker last offered; picking from it chooses a Redux model where Redux is offered. */
let offeredModelTypes: readonly string[] = [];
const ADAPTER = {
  gallery: { findImage: vi.fn() },
  models: {
    ModelSelect: ({ modelTypes, onChange }: { modelTypes: string[]; onChange: (model: unknown) => void }) => {
      offeredModelTypes = modelTypes;
      return (
        <button type="button" onClick={() => onChange(modelTypes.includes('flux_redux') ? REDUX : null)}>
          Model picker
        </button>
      );
    },
    getBaseLabel: (base: string) => base,
    openManager,
  },
  notifications: { info: vi.fn(), undoable },
  project: { activeProjectId: 'project-1' },
  settings: {
    patchGenerateSettings: (patch: Partial<GenerateSettings>) => setStored(patch.referenceImages ?? stored),
    readGenerateValues: () => ({
      ...getDefaultGenerateSettings(SDXL),
      model: SDXL,
      modelKey: SDXL.key,
      referenceImages: stored,
    }),
  },
} as unknown as GenerationUiAdapter;

const reference = (id: string, name?: string): GenerateReferenceImage => ({
  config: {
    beginEndStepPct: [0.2, 0.8],
    clipVisionModel: 'ViT-H',
    image: { original: { image: { height: 768, image_name: `${id}.png`, width: 512 } } },
    method: 'composition',
    model: { base: 'sdxl', key: 'ip-adapter-sdxl', name: 'IP Adapter SDXL', type: 'ip_adapter' },
    type: 'ip_adapter',
    weight: 0.8,
  },
  id,
  isEnabled: true,
  ...(name === undefined ? {} : { name }),
});

/** What the section last committed, read back after each interaction. */
let stored: GenerateReferenceImage[] = [];
/** Replaces the rendered references, as a model switch or another tab would. */
const section = createRef<{ replace: (referenceImages: GenerateReferenceImage[]) => void }>();
const setStored = (referenceImages: GenerateReferenceImage[]) => section.current?.replace(referenceImages);

/** Each enabled reference without an adapter blocks Invoke, worded as Generate validation words it. */
const needsAdapterIssues = (referenceImages: readonly GenerateReferenceImage[]): GenerationIssue[] =>
  referenceImages.flatMap((referenceImage, index) =>
    referenceImage.isEnabled && referenceImage.config.type === 'ip_adapter' && referenceImage.config.model === null
      ? [
          {
            fix: { kind: 'turnOffReferenceImages', referenceImageIds: [referenceImage.id] },
            id: `referenceImages:needsIpAdapter:${referenceImage.id}`,
            message: `${referenceImage.name ?? `Reference image ${index + 1}`} needs an IP Adapter model for SDXL.`,
            messageKey: 'widgets.generate.issues.referenceImageNeedsIpAdapter',
            messageValues: {
              index: index + 1,
              name: 'SDXL',
              ...(referenceImage.name === undefined ? {} : { referenceName: referenceImage.name }),
            },
            target: { field: 'referenceImage', referenceImageId: referenceImage.id, section: 'guidance' },
          } satisfies GenerationIssue,
        ]
      : []
  );

/** The form applies each reference update to its draft at once; this keeps that draft, and validates it. */
const Section = ({
  initial,
  model,
  models,
  ref,
}: {
  initial: GenerateReferenceImage[];
  model: GenerateModelConfig;
  models: readonly GenerationModelCatalogItem[];
  ref: Ref<{ replace: (referenceImages: GenerateReferenceImage[]) => void }>;
}) => {
  const [referenceImages, setReferenceImages] = useState(initial);
  const replace = useCallback((next: GenerateReferenceImage[]) => {
    stored = next;
    setReferenceImages(next);
  }, []);
  useImperativeHandle(ref, () => ({ replace }), [replace]);
  const applyFix = useCallback((fix: GenerationIssueFix) => {
    if (fix.kind === 'turnOffReferenceImages') {
      setReferenceImages((current) => {
        const next = current.map((referenceImage) =>
          fix.referenceImageIds.includes(referenceImage.id) ? { ...referenceImage, isEnabled: false } : referenceImage
        );
        stored = next;
        return next;
      });
    }
  }, []);
  const onCommit = useCallback(
    (update: GenerateSettingsUpdate) => {
      // Applied at once, as the form's draft does, so a caller reading back what it removed finds it.
      const next =
        typeof update === 'function'
          ? update({ referenceImages: stored } as GenerateSettings).referenceImages
          : (update.referenceImages ?? stored);
      replace(next);
    },
    [replace]
  );
  const onCommitImmediate = useCallback(
    (patch: Partial<GenerateSettings>) => (patch.referenceImages ? replace(patch.referenceImages) : undefined),
    [replace]
  );

  return (
    <GenerateIssuesProvider applyFix={applyFix} issues={needsAdapterIssues(referenceImages)}>
      <GenerateReferenceImagesContent
        models={models}
        referenceImages={referenceImages}
        selectedModel={model}
        visionComponents={NO_VISION_COMPONENTS}
        onCommit={onCommit}
        onCommitImmediate={onCommitImmediate}
      />
    </GenerateIssuesProvider>
  );
};

let host: HTMLDivElement;
let root: Root;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (
  initial: GenerateReferenceImage[],
  model: GenerateModelConfig = SDXL,
  models: readonly GenerationModelCatalogItem[] = NO_MODELS
) => {
  stored = initial;
  await act(() =>
    root.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <GenerationUiProvider adapter={ADAPTER}>
            <DndContext>
              <div data-hotkey-widget-type-id="generate" data-testid="section">
                <Section ref={section} initial={initial} model={model} models={models} />
              </div>
            </DndContext>
          </GenerationUiProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

const list = () => host.querySelector<HTMLElement>('[role="list"][aria-label="Reference images"]')!;
const row = (id: string) => host.querySelector<HTMLElement>(`[data-reference-image-id="${id}"]`)!;
const rowButton = (id: string) => row(id).querySelector<HTMLElement>('[data-list-primary]')!;
const nameOf = (id: string) => row(id).querySelector('[data-list-primary] [title]')?.textContent;
const renderedOrder = () =>
  [...host.querySelectorAll<HTMLElement>('[data-reference-image-id]')].map(
    (element) => element.dataset.referenceImageId
  );
const nameField = () => host.querySelector<HTMLInputElement>('input[aria-label="Rename"]');
const addRow = () => host.querySelector<HTMLButtonElement>('[data-reference-add]')!;
const menuItem = (value: string) => document.querySelector<HTMLElement>(`[role="menuitem"][data-value="${value}"]`);
const pressOn = async (element: HTMLElement, keys: string) => {
  await act(() => element.focus());
  await act(() => userEvent.keyboard(keys));
};

beforeEach(() => {
  undoable.mockClear();
  host = document.createElement('div');
  host.style.width = '360px';
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  await userEvent.unhover(document.body);
});

describe('reference image rows', () => {
  it('lists each reference as a List row, with dividers and the add row as the last row', async () => {
    await render([reference('a', 'Hero pose'), reference('b')]);

    const items = [...list().children].filter((child) => child.getAttribute('role') === 'listitem');
    expect(items).toHaveLength(3);
    // Hairlines between rows, the add row included.
    expect(list().querySelectorAll(':scope > [data-list-divider]')).toHaveLength(2);
    expect(items[2]?.contains(addRow())).toBe(true);
    expect(addRow().hasAttribute('data-list-surface')).toBe(true);

    // Each row's controls are named in its reference's context, and its summary reads its settings.
    expect(row('a').querySelector('[role="group"]')?.getAttribute('aria-label')).toBe('Hero pose');
    expect(row('b').querySelector('[role="group"]')?.getAttribute('aria-label')).toBe('Reference image 2');
    expect(rowButton('a').getAttribute('aria-haspopup')).toBe('menu');
    expect(rowButton('a').textContent).toContain('Composition · 20–80% · IP Adapter SDXL');
    expect(row('a').querySelector('[data-scope="switch"]')?.textContent).toBe('Disable Hero pose');
    expect(row('a').querySelector('button[aria-label="More actions for Hero pose"]')).not.toBeNull();
    // The weight is on the row; the rest waits under the disclosure.
    expect(row('a').querySelector('[role="slider"]')?.getAttribute('aria-valuenow')).toBe('0.8');
    expect(row('a').querySelector('[aria-label="Mode"]')).toBeNull();
  });

  it('is the add row alone when empty, counting the slots', async () => {
    await render([]);

    expect(list().querySelectorAll('[role="listitem"]')).toHaveLength(1);
    expect(addRow().textContent).toContain('Add reference image');
    expect(addRow().textContent).toContain(`0 of ${getMaxReferenceImages(SDXL)} used`);
    expect(addRow().getAttribute('aria-disabled')).toBeNull();
  });

  it('opens and closes the settings from the chevron and from Right and Left on the row', async () => {
    await render([reference('a')]);
    const chevron = row('a').querySelector<HTMLButtonElement>('button[aria-label="Reference image 1 settings"]')!;

    expect(chevron.getAttribute('aria-expanded')).toBe('false');
    await act(() => chevron.click());
    expect(chevron.getAttribute('aria-expanded')).toBe('true');
    expect(document.getElementById(chevron.getAttribute('aria-controls')!)).not.toBeNull();
    expect(row('a').querySelector('[aria-label="Mode"]')).not.toBeNull();
    expect(row('a').textContent).toContain('ViT-H');
    expect(row('a').querySelector('button[aria-label="Crop Reference image 1"]')).not.toBeNull();
    // The one level of disclosure: no nested Advanced.
    expect(row('a').textContent).not.toContain('Advanced');

    await pressOn(rowButton('a'), '{ArrowLeft}');
    expect(chevron.getAttribute('aria-expanded')).toBe('false');
    expect(row('a').querySelector('[aria-label="Mode"]')).toBeNull();

    await pressOn(rowButton('a'), '{ArrowRight}');
    expect(chevron.getAttribute('aria-expanded')).toBe('true');
    expect(document.activeElement).toBe(rowButton('a'));
  });

  it('updates the summary live as the step window changes', async () => {
    await render([reference('a')]);
    await act(() =>
      row('a').querySelector<HTMLButtonElement>('button[aria-label="Reference image 1 settings"]')!.click()
    );

    const [begin] = [...row('a').querySelectorAll<HTMLElement>('[role="slider"][aria-label="Active steps"]')];
    await pressOn(begin!, '{ArrowLeft}');

    expect(stored[0]?.config).toMatchObject({ beginEndStepPct: [0.15, 0.8] });
    expect(rowButton('a').textContent).toContain('Composition · 15–80% · IP Adapter SDXL');
    expect(row('a').textContent).toContain('15–80%');
  });
});

describe('renaming a reference', () => {
  it('renames on a double-click, committing with Enter and handing focus back to the row', async () => {
    await render([reference('a'), reference('b')]);

    await act(() => userEvent.dblClick(rowButton('a')));
    const field = nameField()!;

    expect(field).not.toBeNull();
    expect(document.activeElement).toBe(field);
    // The whole name is selected, ready to be replaced.
    expect([field.selectionStart, field.selectionEnd]).toEqual([0, 'Reference image 1'.length]);
    expect(field.closest('button')).toBeNull();

    await act(() => userEvent.keyboard('  Hero pose  {Enter}'));

    expect(nameField()).toBeNull();
    expect(nameOf('a')).toBe('Hero pose');
    expect(stored[0]?.name).toBe('Hero pose');
    await expect.poll(() => document.activeElement).toBe(rowButton('a'));
  });

  it('renames with F2, cancels with Escape, and commits on blur', async () => {
    await render([reference('a', 'Hero pose')]);

    await pressOn(rowButton('a'), '{F2}');
    await act(() => userEvent.keyboard('Something else{Escape}'));

    expect(nameField()).toBeNull();
    expect(nameOf('a')).toBe('Hero pose');
    expect(stored[0]?.name).toBe('Hero pose');
    await expect.poll(() => document.activeElement).toBe(rowButton('a'));

    await pressOn(rowButton('a'), '{F2}');
    await act(() => userEvent.keyboard('Palette'));
    await act(() => addRow().focus());

    expect(nameField()).toBeNull();
    expect(stored[0]?.name).toBe('Palette');
    // Focus stays where it went.
    expect(document.activeElement).toBe(addRow());
  });

  it('renames from the row menu', async () => {
    await render([reference('a')]);

    await act(() =>
      userEvent.click(row('a').querySelector<HTMLElement>('button[aria-label="More actions for Reference image 1"]')!)
    );
    await expect.poll(() => menuItem('rename')).not.toBeNull();
    await act(() => userEvent.click(menuItem('rename')!));

    // The closing menu reclaims focus for its exit animation; the field must keep it.
    await expect.poll(() => document.activeElement).toBe(nameField());
    await act(() => userEvent.keyboard('Lighting{Enter}'));

    expect(stored[0]?.name).toBe('Lighting');
  });

  it("stores no name for an empty name or the row's own default", async () => {
    await render([reference('a', 'Hero pose'), reference('b', 'Palette')]);

    await pressOn(rowButton('a'), '{F2}');
    await act(() => userEvent.keyboard('{Backspace}{Enter}'));
    await pressOn(rowButton('b'), '{F2}');
    await act(() => userEvent.keyboard('reference image 2{Enter}'));

    expect([nameOf('a'), nameOf('b')]).toEqual(['Reference image 1', 'Reference image 2']);
    expect(stored.map((referenceImage) => 'name' in referenceImage)).toEqual([false, false]);
  });

  it("keeps another position's default in the field with its reason, never changing it unseen", async () => {
    await render([reference('a', 'Hero pose'), reference('b'), reference('c')]);

    await pressOn(rowButton('a'), '{F2}');
    await act(() => userEvent.keyboard('Reference image 3'));
    const reason = "Reference image 3 is another reference's default name. Choose a different one.";

    // Said as it is typed, tied to the field, and Enter keeps the field open on it.
    expect(row('a').querySelector('[role="alert"]')?.textContent).toBe(reason);
    expect(nameField()?.getAttribute('aria-invalid')).toBe('true');
    await act(() => userEvent.keyboard('{Enter}'));
    expect(nameField()?.value).toBe('Reference image 3');
    expect(stored[0]?.name).toBe('Hero pose');

    // A name that no longer reads as a default clears the reason and is kept.
    await act(() => userEvent.keyboard(' close-up{Enter}'));
    expect(nameField()).toBeNull();
    expect(stored[0]?.name).toBe('Reference image 3 close-up');
    expect(row('a').querySelector('[role="alert"]')).toBeNull();

    // Leaving the field on a reserved name keeps the name it had.
    await pressOn(rowButton('b'), '{F2}');
    await act(() => userEvent.keyboard('reference image 1'));
    await act(() => addRow().focus());
    expect(nameField()).toBeNull();
    expect(nameOf('b')).toBe('Reference image 2');
    expect('name' in stored[1]!).toBe(false);
  });

  it('leaves Enter and Escape to an IME while it composes, and keeps Escape from reaching the panel', async () => {
    const panelKeys = vi.fn();
    document.body.addEventListener('keydown', panelKeys);
    await render([reference('a', 'Hero pose')]);

    await pressOn(rowButton('a'), '{F2}');
    await act(async () => {
      await commands.imeCompose([{ kind: 'compose', text: 'にほん' }]);
    });
    // The keys that confirm or drop a candidate arrive marked as composing.
    for (const key of ['Enter', 'Escape']) {
      await act(() =>
        nameField()!.dispatchEvent(
          new KeyboardEvent('keydown', { bubbles: true, cancelable: true, isComposing: true, key, keyCode: 229 })
        )
      );
    }
    expect(nameField()).not.toBeNull();

    await act(async () => {
      await commands.imeCompose([{ kind: 'commit', text: '日本' }]);
    });
    await act(() => userEvent.keyboard('{Escape}'));

    expect(nameField()).toBeNull();
    expect(stored[0]?.name).toBe('Hero pose');
    expect(panelKeys).not.toHaveBeenCalledWith(expect.objectContaining({ key: 'Escape' }));
    document.body.removeEventListener('keydown', panelKeys);

    await pressOn(rowButton('a'), '{F2}');
    await act(async () => {
      await commands.imeCompose([{ kind: 'compose', text: 'にほん' }]);
      await commands.imeCompose([{ kind: 'commit', text: '日本' }]);
    });
    await act(() => userEvent.keyboard('{Enter}'));
    expect(stored[0]?.name).toBe('日本');
  });

  it('keeps names with their images through a reorder and the removal of a sibling', async () => {
    await render([reference('a', 'Hero pose'), reference('b'), reference('c', 'Lighting')]);

    await pressOn(rowButton('a'), '{Alt>}{ArrowDown}{/Alt}');
    expect(renderedOrder()).toEqual(['b', 'a', 'c']);
    expect([nameOf('b'), nameOf('a'), nameOf('c')]).toEqual(['Reference image 1', 'Hero pose', 'Lighting']);

    await pressOn(rowButton('b'), '{Delete}');
    expect(renderedOrder()).toEqual(['a', 'c']);
    expect([nameOf('a'), nameOf('c')]).toEqual(['Hero pose', 'Lighting']);
    expect(stored.map(({ id, name }) => [id, name])).toEqual([
      ['a', 'Hero pose'],
      ['c', 'Lighting'],
    ]);
  });
});

describe('reordering and removing references', () => {
  it('moves with Alt+Up and Alt+Down, focus staying on the moved row, and stops at the ends', async () => {
    await render([reference('a'), reference('b'), reference('c')]);

    await pressOn(rowButton('c'), '{Alt>}{ArrowUp}{/Alt}');
    expect(renderedOrder()).toEqual(['a', 'c', 'b']);
    await expect.poll(() => document.activeElement).toBe(rowButton('c'));

    await act(() => userEvent.keyboard('{Alt>}{ArrowUp}{ArrowUp}{/Alt}'));
    expect(renderedOrder()).toEqual(['c', 'a', 'b']);
    await expect.poll(() => document.activeElement).toBe(rowButton('c'));
  });

  it('offers Move up and Move down in the row menu, disabled at the ends', async () => {
    await render([reference('a'), reference('b')]);

    await act(() =>
      userEvent.click(row('a').querySelector<HTMLElement>('button[aria-haspopup="menu"]:not([data-list-primary])')!)
    );
    await expect.poll(() => menuItem('move-up')).not.toBeNull();
    expect(menuItem('move-up')?.hasAttribute('data-disabled')).toBe(true);
    expect(menuItem('move-down')?.hasAttribute('data-disabled')).toBe(false);

    await act(() => userEvent.click(menuItem('move-down')!));
    expect(renderedOrder()).toEqual(['b', 'a']);
  });

  it('removes with Delete only, one row for a held key, and Undo puts it back in place as it was', async () => {
    const cropped = {
      ...reference('b', 'Palette'),
      config: {
        ...reference('b').config,
        image: {
          crop: {
            box: { height: 64, width: 64, x: 0, y: 0 },
            image: { height: 64, image_name: 'b-crop.png', width: 64 },
            ratio: null,
          },
          original: { image: { height: 768, image_name: 'b.png', width: 512 } },
        },
      },
    } as GenerateReferenceImage;
    await render([reference('a'), cropped, reference('c')]);

    // Backspace is for text: on a row it removes nothing.
    await pressOn(rowButton('b'), '{Backspace}');
    expect(renderedOrder()).toEqual(['a', 'b', 'c']);

    // Held, Delete repeats; only its first press removes.
    await pressOn(rowButton('b'), '{Delete>4/}');
    expect(renderedOrder()).toEqual(['a', 'c']);
    expect(undoable).toHaveBeenCalledOnce();
    expect(undoable.mock.calls[0]![0]).toBe('Removed Palette');

    await act(() => undoable.mock.calls[0]![1].onUndo());
    expect(renderedOrder()).toEqual(['a', 'b', 'c']);
    expect(stored[1]).toEqual(cropped);
    await expect.poll(() => document.activeElement).toBe(rowButton('b'));
  });

  it('removes from the row menu, undoably', async () => {
    await render([reference('a'), reference('b')]);

    await act(() =>
      userEvent.click(row('a').querySelector<HTMLElement>('button[aria-haspopup="menu"]:not([data-list-primary])')!)
    );
    await expect.poll(() => menuItem('remove')).not.toBeNull();
    await act(() => userEvent.click(menuItem('remove')!));
    expect(renderedOrder()).toEqual(['b']);
    expect(undoable.mock.calls[0]![0]).toBe('Removed Reference image 1');

    await act(() => undoable.mock.calls[0]![1].onUndo());
    expect(renderedOrder()).toEqual(['a', 'b']);
  });

  it('removes with Delete, focusing the next row, else the previous one, else the add row', async () => {
    await render([reference('a'), reference('b'), reference('c')]);

    await pressOn(rowButton('b'), '{Delete}');
    expect(renderedOrder()).toEqual(['a', 'c']);
    await expect.poll(() => document.activeElement).toBe(rowButton('c'));

    await act(() => userEvent.keyboard('{Delete}'));
    await expect.poll(() => document.activeElement).toBe(rowButton('a'));

    await act(() => userEvent.keyboard('{Delete}'));
    expect(renderedOrder()).toEqual([]);
    await expect.poll(() => document.activeElement).toBe(addRow());
  });
});

describe('references the model cannot use', () => {
  it('stay listed but inert, badged, with Remove all above them', async () => {
    await render([reference('a', 'Hero pose')], COGVIEW);

    const button = rowButton('a');
    expect(row('a').textContent).toContain('Incompatible');
    expect(button.getAttribute('aria-disabled')).toBe('true');
    expect(row('a').querySelector('[inert]')).not.toBeNull();
    expect(host.textContent).toContain('Remove all');
    expect(host.querySelector('[data-reference-add]')).toBeNull();

    // Renaming, moving and removing all wait for a model that takes references.
    await act(() => button.dispatchEvent(new MouseEvent('dblclick', { bubbles: true })));
    await pressOn(button, '{F2}');
    await pressOn(button, '{Delete}');
    expect(nameField()).toBeNull();
    expect(stored).toHaveLength(1);
  });
});

describe('a narrow Generate panel', () => {
  it('keeps every control of three references, one of them open, inside 350px', async () => {
    // The Generate panel at its narrowest, with the section's own padding the rows reach into.
    host.style.cssText = 'box-sizing:border-box;padding:0 8px;width:350px;';
    await render([
      reference('a', 'A name long enough to need truncating at this width'),
      reference('b'),
      reference('c'),
    ]);
    await act(() => row('a').querySelector<HTMLButtonElement>('button[aria-label^="A name long"]')!.click());

    const bounds = host.getBoundingClientRect();
    const controls = [
      ...host.querySelectorAll<HTMLElement>(
        'button, [role="slider"], [role="radio"], input:not([type="hidden"]), [role="combobox"]'
      ),
    ]
      // The changed-value dot sits in the gutter left of its field, as on a concept row, reaching into the panel's edge.
      .filter((element) => element.getClientRects().length > 0 && !element.hasAttribute('data-modified-dot'));

    expect(controls.length).toBeGreaterThan(10);
    for (const control of controls) {
      const rect = control.getBoundingClientRect();
      expect(rect.left).toBeGreaterThanOrEqual(bounds.left - 0.5);
      expect(rect.right).toBeLessThanOrEqual(bounds.right + 0.5);
    }
    expect(list().scrollWidth).toBeLessThanOrEqual(list().clientWidth);
  });
});

describe('a reference that blocks Invoke', () => {
  const unadapted = (id: string): GenerateReferenceImage => {
    const referenceImage = reference(id);
    return referenceImage.config.type === 'ip_adapter'
      ? { ...referenceImage, config: { ...referenceImage.config, model: null } }
      : referenceImage;
  };
  const issueIcon = () => row('a').querySelector<HTMLButtonElement>('button[data-issue-ids]');
  const fixButton = () => document.querySelector<HTMLButtonElement>('[data-issue-id] [data-issue-fix]');

  it('turns off from the issue popover, which resolves the issue and hands focus to the switch', async () => {
    await render([unadapted('a')]);
    // The row grows no line for its issue.
    expect(row('a').querySelector('[data-issue-notes]')).toBeNull();

    await act(() => userEvent.click(issueIcon()!));
    await expect.poll(() => fixButton()?.textContent).toBe('Turn off');
    await act(() => userEvent.click(fixButton()!));

    expect(stored[0]?.isEnabled).toBe(false);
    expect(issueIcon()).toBeNull();
    await expect.poll(() => fixButton()).toBeNull();
    await expect.poll(() => document.activeElement).toBe(row('a').querySelector('[data-scope="switch"] input'));
  });

  it('opens from the Invoke issue list, so the message and its fix show', async () => {
    await render([reference('ok'), unadapted('a')]);
    const [issue] = needsAdapterIssues(stored);

    await act(async () => {
      revealGenerateIssue(issue!);
      await new Promise(requestAnimationFrame);
    });

    await expect.poll(() => fixButton()?.closest('[data-issue-id]')?.textContent).toContain(issue!.message);
    expect(issueIcon()?.getAttribute('aria-expanded')).toBe('true');
  });
});

describe('a reference without its model', () => {
  const unadaptedNamed = (id: string, name?: string): GenerateReferenceImage => {
    const referenceImage = reference(id, name);
    return referenceImage.config.type === 'ip_adapter'
      ? { ...referenceImage, config: { ...referenceImage.config, model: null } }
      : referenceImage;
  };
  const chevron = () => row('a').querySelector<HTMLButtonElement>('button[aria-controls]')!;
  const issueIcon = () => row('a').querySelector<HTMLButtonElement>('button[data-issue-ids]');

  it("names its issue by the reference's name and opens no settings on its own when it loads", async () => {
    await render([unadaptedNamed('a', 'Hero pose')]);

    expect(chevron().getAttribute('aria-expanded')).toBe('false');
    expect(issueIcon()?.getAttribute('aria-label')).toBe('1 issue: Hero pose needs an IP Adapter model for SDXL.');
  });

  it('offers Choose model when a model switch leaves it without one, opening the settings on the picker', async () => {
    await render([reference('a')], SDXL, SDXL_ADAPTERS);
    expect(issueIcon()).toBeNull();

    // A switch to a model whose adapters are not installed leaves the same row without one.
    await act(() => setStored([unadaptedNamed('a')]));
    expect(chevron().getAttribute('aria-expanded')).toBe('false');

    await act(() => userEvent.click(issueIcon()!));
    const action = () => document.querySelector<HTMLButtonElement>('[data-issue-id] [data-issue-action]');
    await expect.poll(() => action()?.textContent).toBe('Choose model');
    expect(document.querySelector('[data-issue-id] [data-issue-fix]')?.textContent).toBe('Turn off');
    await act(() => userEvent.click(action()!));

    expect(chevron().getAttribute('aria-expanded')).toBe('true');
    await expect.poll(() => document.activeElement?.closest('[data-reference-model]')).not.toBeNull();
    expect(stored[0]?.isEnabled).toBe(true);
  });

  it('offers the Model Manager instead when no model it could use is installed', async () => {
    openManager.mockClear();
    await render([unadaptedNamed('a')]);

    await act(() => userEvent.click(issueIcon()!));
    const action = () => document.querySelector<HTMLButtonElement>('[data-issue-id] [data-issue-action]');
    await expect.poll(() => action()?.textContent).toBe('Open Model Manager');
    await act(() => userEvent.click(action()!));

    expect(openManager).toHaveBeenCalledExactlyOnceWith({ modelType: 'ip_adapter' });
    expect(chevron().getAttribute('aria-expanded')).toBe('false');
  });
});

describe('positional default names', () => {
  const translatorFor = async (template: string) => {
    const instance = i18next.createInstance();
    await instance.init({
      interpolation: { escapeValue: false },
      lng: 'en',
      resources: { en: { translation: { widgets: { generate: { referenceImageNumbered: template } } } } },
    });
    return instance.t.bind(instance);
  };

  it('match any position in the locale wording, and give up safely on templates they cannot read', async () => {
    const english = getPositionalDefaultNameTest(await translatorFor('Reference image {{index}}'))!;
    expect(['Reference image 3', ' reference image 12 '].map(english)).toEqual([true, true]);
    expect(['Reference image', 'Reference image 3 close-up', 'Hero pose'].map(english)).toEqual([false, false, false]);

    expect(getPositionalDefaultNameTest(await translatorFor('{{index}}. Referenzbild'))!('2. referenzbild')).toBe(true);
    // A bare number, or a formatter that loses the placeholder, would reserve names it should not: no pattern.
    expect(getPositionalDefaultNameTest(await translatorFor('{{index}}'))).toBeNull();
    expect(getPositionalDefaultNameTest(await translatorFor('Image {{index, number}}'))).toBeNull();
  });
});

describe('a FLUX Kontext reference', () => {
  it('can become a Redux reference by picking a Redux model, as before', async () => {
    const kontext: GenerateReferenceImage = {
      config: {
        image: { original: { image: { height: 512, image_name: 'k.png', width: 512 } } },
        model: null,
        type: 'flux_kontext_reference_image',
      },
      id: 'a',
      isEnabled: true,
    };
    await render([kontext], FLUX);

    expect(offeredModelTypes).toEqual(['ip_adapter', 'flux_redux']);
    await act(() => userEvent.click(row('a').querySelector<HTMLButtonElement>('[data-reference-model] button')!));

    expect(stored[0]?.config).toMatchObject({ imageInfluence: 'highest', model: REDUX, type: 'flux_redux' });
  });
});

describe('row summaries and surfaces', () => {
  const t = i18n.t.bind(i18n);
  const image = { original: { image: { height: 768, image_name: 'x.png', width: 512 } } };
  const crop = {
    box: { height: 300, width: 200, x: 0, y: 0 },
    image: { height: 300, image_name: 'c.png', width: 200 },
    ratio: null,
  };

  it('says what each family does, in the words its controls use', () => {
    expect(describeReferenceImage({ image, imageInfluence: 'low', model: null, type: 'flux_redux' }, false, t)).toBe(
      'No model'
    );
    expect(
      describeReferenceImage(
        {
          image,
          imageInfluence: 'low',
          model: { base: 'flux', key: 'r', name: 'FLUX Redux', type: 'flux_redux' },
          type: 'flux_redux',
        },
        false,
        t
      )
    ).toBe('FLUX Redux');
    expect(describeReferenceImage({ image, model: null, type: 'flux_kontext_reference_image' }, false, t)).toBe(
      '512 × 768'
    );
    expect(describeReferenceImage({ image, type: 'flux2_reference_image' }, false, t)).toBe('512 × 768');
    expect(describeReferenceImage({ image: { ...image, crop }, type: 'flux2_reference_image' }, false, t)).toBe(
      '200 × 300 · cropped'
    );
    expect(describeReferenceImage({ image: null, type: 'flux2_reference_image' }, false, t)).toBe('No image');
  });

  it("backs every control of an open row with its own surface, so the row's hover tint never washes it out", async () => {
    await render([reference('a')]);
    await act(() => row('a').querySelector<HTMLButtonElement>('button[aria-controls]')!.click());

    const controls = [
      ...row('a').querySelectorAll<HTMLElement>(
        '[data-scope="scrubber"], [data-scope="segment-group"][data-part="root"], [data-scope="select"][data-part="trigger"], [data-reference-model] button, [role="slider"][aria-label="Active steps"]'
      ),
    ];

    expect(controls.length).toBeGreaterThanOrEqual(5);
    for (const control of controls) {
      expect(control.closest('[data-control-surface]')).not.toBeNull();
    }
  });
});
