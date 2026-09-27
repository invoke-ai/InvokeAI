import type { FieldInputTemplate } from '@features/workflow/contracts';

import { ChakraProvider, Field } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { getWorkflowFieldInvalidReason } from '@features/workflow/utility';
import { toaster } from '@platform/ui';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act, cloneElement, startTransition, useCallback, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { server, userEvent } from 'vitest/browser';

import { WorkflowFieldInput, type WorkflowFieldInputProps } from './WorkflowFieldInput';

declare module 'vitest/browser' {
  interface BrowserCommands {
    imeCompose: (steps: readonly { kind: 'commit' | 'compose'; text: string }[]) => Promise<void>;
  }
}

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const uploadImageMock = vi.fn();
const uploadVideoMock = vi.fn();
const resolveItemMock = vi.fn();

const pickerState = vi.hoisted(() => ({ accept: null as readonly string[] | null }));
const workflowApiMock = vi.hoisted(() => ({ apiFetch: vi.fn(), apiFetchJson: vi.fn() }));
const workflowCommandsMock = vi.hoisted(() => ({ editGraph: vi.fn() }));

vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  apiFetch: workflowApiMock.apiFetch,
  apiFetchJson: workflowApiMock.apiFetchJson,
}));

// Stub Gallery transport while asserting accepted kinds and returned items at the field boundary.
vi.mock('@features/gallery/picker', () => ({
  GalleryPickerPopover: ({
    accept,
    children,
    onPick,
  }: {
    accept: readonly string[];
    children: React.ReactElement<{ onClick?: () => void }>;
    onPick: (item: unknown) => void;
  }) => {
    pickerState.accept = accept;

    return cloneElement(children, { onClick: () => onPick(SELECTED_GALLERY_VIDEO) });
  },
}));

vi.mock('@features/gallery', () => ({
  formatGalleryVideoDuration: (seconds: number) => `${seconds}s`,
  galleryDestinations: { list: () => Promise.resolve([]) },
  galleryItems: { resolve: (...args: unknown[]) => resolveItemMock(...args) },
  galleryTransfers: {
    upload: (...args: unknown[]) => uploadImageMock(...args),
    uploadVideo: (...args: unknown[]) => uploadVideoMock(...args),
  },
}));

const modelSelectState = vi.hoisted(() => ({
  props: null as null | {
    excludeKeys?: ReadonlySet<string>;
    filter?: (model: Record<string, unknown>) => boolean;
    modelTypes: readonly string[];
    onChange: (model: unknown) => void;
    placeholder?: string;
  },
}));

// Stub model-library UI while asserting picker filters, exclusions, and selected models.
vi.mock('@features/models/react', () => ({
  ModelSelect: (props: NonNullable<typeof modelSelectState.props>) => {
    modelSelectState.props = props;

    return (
      <button type="button" onClick={() => props.onChange(ADDABLE_LORA)}>
        {props.placeholder}
      </button>
    );
  },
}));

const SD_LORA = {
  base: 'sd-1',
  default_settings: { weight: 0.4 },
  hash: 'hash-sd',
  key: 'sd-lora',
  name: 'Detail Tweaker',
  type: 'lora',
};
const SDXL_LORA = { base: 'sdxl', hash: 'hash-sdxl', key: 'sdxl-lora', name: 'Pixel Art XL', type: 'lora' };
// What the real picker would hand back for this template: an sd-1 LoRA carrying a model default
// weight, so "adds at the model default" cannot pass by coincidence.
const ADDABLE_LORA = {
  base: 'sd-1',
  default_settings: { weight: 0.4 },
  hash: 'hash-add',
  key: 'add-lora',
  name: 'Add Me',
  type: 'lora',
};

const galleryValues: Record<string, unknown> = {};
const graphNodes: unknown[] = [];
const graphEdges: unknown[] = [];
const workflowValues: Record<string, unknown> = {};
const projectSnapshot = {
  galleryValues,
  id: 'project-1',
  projectGraph: { edges: graphEdges, nodes: graphNodes },
  workflowValues,
};

vi.mock('@features/workflow/ui/WorkflowUiContext', () => ({
  useWorkflowProjectSelector: (selector: (project: typeof projectSnapshot) => unknown) => selector(projectSnapshot),
  useWorkflowUi: () => ({ commands: workflowCommandsMock, project: { getSnapshot: () => projectSnapshot } }),
}));

const TEXTAREA_TEMPLATE = {
  name: 'prompt',
  title: 'Prompt',
  type: { name: 'StringField' },
  uiComponent: 'textarea',
} as unknown as FieldInputTemplate;

const VIDEO_TEMPLATE = {
  name: 'video',
  title: 'Video',
  type: { name: 'VideoField' },
} as unknown as FieldInputTemplate;

const SEED_TEMPLATE = {
  exclusiveMaximum: null,
  exclusiveMinimum: null,
  input: 'any',
  maximum: 4_294_967_295,
  minimum: 0,
  multipleOf: null,
  name: 'seed',
  title: 'Seed',
  type: { cardinality: 'SINGLE', name: 'IntegerField' },
} as unknown as FieldInputTemplate;

const FRAME_INDEX_TEMPLATE = {
  name: 'frame_index',
  title: 'Frame Index',
  type: { cardinality: 'SINGLE', name: 'IntegerField' },
  uiComponent: 'video-frame-index',
} as unknown as FieldInputTemplate;

const SAVED_WORKFLOW_TEMPLATE = {
  input: 'direct',
  name: 'workflow_id',
  title: 'Workflow',
  type: { batch: false, cardinality: 'SINGLE', name: 'SavedWorkflowField' },
} as unknown as FieldInputTemplate;

const NUMERIC_ENUM_TEMPLATE = {
  name: 'max_seq_len',
  options: [256, 512],
  title: 'Maximum sequence length',
  type: { batch: false, cardinality: 'SINGLE', name: 'EnumField' },
} as unknown as FieldInputTemplate;

const makeFrameNode = (videoValue: { video_name: string } | undefined) => ({
  data: {
    inputs: {
      frame_index: { label: '', name: 'frame_index', value: -1 },
      video: { label: '', name: 'video', value: videoValue },
    },
    type: 'video_frame_extract',
  },
  id: 'frame-node',
  type: 'invocation',
});

const LORA_COLLECTION_TEMPLATE = {
  name: 'loras',
  title: 'LoRAs',
  type: { cardinality: 'SINGLE_OR_COLLECTION', name: 'LoRAField' },
  uiModelBase: ['sd-1', 'sd-2'],
  uiModelType: ['lora'],
} as unknown as FieldInputTemplate;

const loraEntry = (overrides: Record<string, unknown> = {}) => ({
  lora: { base: 'sd-1', hash: 'hash-sd', key: 'sd-lora', name: 'Detail Tweaker', type: 'lora' },
  weight: 0.75,
  ...overrides,
});

const SELECTED_GALLERY_VIDEO = {
  boardId: 'none',
  category: 'general',
  createdAt: '2026-01-01T00:00:00Z',
  durationSeconds: 5,
  fullUrl: '/api/v1/videos/i/clip.mp4/full',
  height: 480,
  isIntermediate: false,
  kind: 'video',
  name: 'clip.mp4',
  starred: false,
  thumbnailUrl: '/api/v1/videos/i/clip.mp4/thumbnail',
  width: 640,
};

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  uploadImageMock.mockReset();
  uploadVideoMock.mockReset();
  resolveItemMock.mockReset();
  resolveItemMock.mockResolvedValue(SELECTED_GALLERY_VIDEO);
  workflowApiMock.apiFetch.mockReset().mockResolvedValue(new Response());
  workflowApiMock.apiFetchJson.mockReset();
  workflowCommandsMock.editGraph.mockReset();
  // Module-level capture: without this the next test's wait is satisfied by the previous test's
  // props and asserts against a picker that is no longer mounted.
  modelSelectState.props = null;
  queryClient.clear();
  delete galleryValues.selectedImage;
  graphNodes.length = 0;
  graphEdges.length = 0;
  delete workflowValues.batchCount;
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

// No retry backoff: failure states must be observable inside a test timeout.
const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });

const renderField = async (
  template: FieldInputTemplate,
  value: unknown,
  onChange: (value: unknown) => void,
  nodeId?: string,
  seedProps: Pick<WorkflowFieldInputProps, 'onSeedModeChange' | 'seedMode'> = {}
) => {
  await act(() => {
    root.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient}>
          <DndContext>
            <WorkflowFieldInput nodeId={nodeId} template={template} value={value} onChange={onChange} {...seedProps} />
          </DndContext>
        </QueryClientProvider>
      </ChakraProvider>
    );
  });
};

/** Real keystrokes: the weight regressions live between inputs, where a direct DOM write never goes. */
const typeWeight = async (input: HTMLInputElement, keys: string) => {
  await act(async () => {
    await userEvent.click(input);
    await userEvent.keyboard(`{Control>}a{/Control}${keys}`);
  });
};

const pressKey = async (key: string) => {
  await act(async () => {
    await userEvent.keyboard(key);
  });
};

const blurWeight = async (input: HTMLInputElement) => {
  await act(() => {
    input.blur();
  });
};

const findButton = (label: string): HTMLButtonElement => {
  const button = Array.from(host.querySelectorAll('button')).find((el) => el.textContent?.includes(label));

  if (!button) {
    throw new Error(`Button "${label}" not found`);
  }

  return button;
};

const BOOLEAN_TEMPLATE = {
  name: 'enabled',
  title: 'Enabled',
  type: { batch: false, cardinality: 'SINGLE', name: 'BooleanField' },
} as unknown as FieldInputTemplate;

describe('WorkflowFieldInput boolean', () => {
  it('toggles from a click on the switch when hosted in a Field with an external id', async () => {
    const onChange = vi.fn();

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <Field.Root>
            <WorkflowFieldInput id="node-enabled-value" template={BOOLEAN_TEMPLATE} value={false} onChange={onChange} />
          </Field.Root>
        </ChakraProvider>
      );
    });

    const control = host.querySelector('[data-scope="switch"][data-part="control"]');

    if (!(control instanceof HTMLElement)) {
      throw new Error('Switch control not rendered');
    }

    await act(async () => {
      await userEvent.click(control);
    });

    expect(onChange).toHaveBeenCalledWith(true);
  });
});

describe('WorkflowFieldInput textarea', () => {
  it('uses the accessible unbounded resizable textarea for prompt-like string fields', async () => {
    await renderField(TEXTAREA_TEMPLATE, 'hello', vi.fn());

    const textarea = host.querySelector<HTMLTextAreaElement>('textarea')!;
    const handle = host.querySelector<HTMLElement>('[role="separator"]')!;

    expect(getComputedStyle(textarea).height).toBe('96px');
    expect(getComputedStyle(textarea).fontFamily).toContain('monospace');
    expect(handle.getAttribute('aria-label')).toBe('Resize Prompt');
    expect(handle.getAttribute('aria-valuemin')).toBe('56');
    expect(handle.hasAttribute('aria-valuemax')).toBe(false);

    await act(() => handle.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowDown' })));
    expect(getComputedStyle(textarea).height).toBe('108px');
  });
});

describe('WorkflowFieldInput typed enums', () => {
  it.each([
    { label: 'numeric', options: [256, 512], selected: 512, next: 256 },
    { label: 'boolean', options: [false, true], selected: false, next: true },
    { label: 'string', options: ['small', 'large'], selected: 'large', next: 'small' },
  ])('shows the selected $label value and preserves its type when changed', async ({ options, selected, next }) => {
    const onChange = vi.fn();
    await renderField({ ...NUMERIC_ENUM_TEMPLATE, options }, selected, onChange);

    const trigger = host.querySelector<HTMLElement>('[data-scope="select"][data-part="trigger"]');
    expect(trigger?.textContent).toContain(String(selected));

    if (!trigger) {
      throw new Error('Numeric enum trigger not rendered');
    }
    await act(() => userEvent.click(trigger));

    const option = Array.from(document.querySelectorAll<HTMLElement>('[data-scope="select"][data-part="item"]')).find(
      (item) => item.textContent?.includes(String(next))
    );
    if (!option) {
      throw new Error('Numeric enum option not rendered');
    }
    await act(() => userEvent.click(option));
    expect(onChange).toHaveBeenCalledWith(next);
  });
});

describe('WorkflowFieldInput saved workflows', () => {
  it('retries a failed child when its already-selected workflow is picked again', async () => {
    const selectedWorkflow = {
      category: 'user',
      call_saved_workflow_compatibility: { is_callable: true, message: null, reason: 'ok' },
      description: '',
      is_public: false,
      name: 'Selected child',
      workflow_id: 'selected-child',
    };
    workflowApiMock.apiFetchJson.mockImplementation((path: string) =>
      path.includes('/i/')
        ? Promise.reject(new Error('Temporary detail failure'))
        : Promise.resolve({ items: [selectedWorkflow], page: 0, pages: 1, total: 1 })
    );
    const onChange = vi.fn();
    await renderField(SAVED_WORKFLOW_TEMPLATE, 'selected-child', onChange, 'call-node');
    await vi.waitFor(() =>
      expect(host.querySelector<HTMLInputElement>('input[role="combobox"]')?.value).toBe('Selected child')
    );
    await vi.waitFor(() =>
      expect(host.querySelector<HTMLButtonElement>('button[aria-label="common.retry"]')).not.toBeNull()
    );

    const input = host.querySelector<HTMLInputElement>('input[role="combobox"]');
    if (!input) {
      throw new Error('Saved workflow combobox not rendered');
    }
    await act(() => userEvent.click(input));
    const selectedItem = Array.from(
      document.querySelectorAll<HTMLElement>('[data-scope="combobox"][data-part="item"]')
    ).find((item) => item.textContent?.includes('Selected child'));
    if (!selectedItem) {
      throw new Error('Selected workflow option not rendered');
    }
    await act(() => userEvent.click(selectedItem));

    expect(workflowCommandsMock.editGraph).toHaveBeenCalledWith({
      nodeId: 'call-node',
      type: 'retryCallSavedWorkflow',
    });
    expect(workflowCommandsMock.editGraph).toHaveBeenCalledTimes(1);
    expect(onChange).not.toHaveBeenCalled();

    workflowCommandsMock.editGraph.mockClear();
    const retryButton = host.querySelector<HTMLButtonElement>('button[aria-label="common.retry"]');
    if (!retryButton) {
      throw new Error('Saved workflow retry button not rendered');
    }
    await act(() => userEvent.click(retryButton));
    expect(workflowCommandsMock.editGraph).toHaveBeenCalledOnce();

    workflowCommandsMock.editGraph.mockClear();
    await act(() => userEvent.click(input));
    await act(() => userEvent.keyboard('{ArrowDown}{Enter}'));
    expect(workflowCommandsMock.editGraph).toHaveBeenCalledOnce();
  });

  it('uses plural loading copy while the workflow list is loading', async () => {
    workflowApiMock.apiFetchJson.mockImplementation(() => new Promise(() => {}));

    await renderField(SAVED_WORKFLOW_TEMPLATE, '', vi.fn());

    await vi.waitFor(() =>
      expect(host.querySelector<HTMLInputElement>('input[role="combobox"]')?.placeholder).toBe(
        'nodes.savedWorkflowListLoading'
      )
    );
  });

  it('displays dynamic workflow names and marks incompatible workflows disabled', async () => {
    const onChange = vi.fn();
    workflowApiMock.apiFetchJson.mockImplementation((path: string) => {
      if (path.includes('is_public=true')) {
        return Promise.resolve({
          items: [
            {
              category: 'user',
              call_saved_workflow_compatibility: { is_callable: true, message: null, reason: 'ok' },
              description: '',
              is_public: true,
              name: 'Shared Dynamic Workflow',
              workflow_id: 'shared-workflow',
            },
          ],
          page: 0,
          pages: 1,
          total: 1,
        });
      }

      return Promise.resolve({
        items: [
          {
            category: 'default',
            call_saved_workflow_compatibility: {
              is_callable: false,
              message: 'Missing workflow return node',
              reason: 'missing_workflow_return',
            },
            description: '',
            is_public: true,
            name: 'Unsupported Dynamic Workflow',
            workflow_id: 'unsupported-workflow',
          },
        ],
        page: 0,
        pages: 1,
        total: 1,
      });
    });

    await renderField(SAVED_WORKFLOW_TEMPLATE, '', onChange);
    await vi.waitFor(() => expect(workflowApiMock.apiFetchJson).toHaveBeenCalledTimes(2));
    await vi.waitFor(() =>
      expect(host.querySelector<HTMLInputElement>('input[role="combobox"]')?.placeholder).toBe(
        'nodes.savedWorkflowSearch'
      )
    );
    const input = host.querySelector<HTMLInputElement>('input[role="combobox"]');

    if (!input) {
      throw new Error('Saved workflow combobox not rendered');
    }

    await act(async () => {
      await userEvent.click(input);
    });

    await vi.waitFor(() => {
      expect(document.body.textContent).toContain('Shared Dynamic Workflow');
      expect(document.body.textContent).toContain('Unsupported Dynamic Workflow');
    });

    const unsupportedItem = Array.from(document.querySelectorAll('[data-scope="combobox"][data-part="item"]')).find(
      (item) => item.textContent?.includes('Unsupported Dynamic Workflow')
    );

    expect(unsupportedItem?.hasAttribute('data-disabled')).toBe(true);

    const sharedItem = Array.from(document.querySelectorAll('[data-scope="combobox"][data-part="item"]')).find((item) =>
      item.textContent?.includes('Shared Dynamic Workflow')
    );

    if (!(sharedItem instanceof HTMLElement)) {
      throw new Error('Shared workflow option not rendered');
    }

    await act(async () => {
      await userEvent.click(sharedItem);
    });

    expect(onChange).toHaveBeenCalledWith('shared-workflow');
  });
});

describe('WorkflowFieldInput media inputs', () => {
  it('renders media controls when the host provides the workflow dnd context', async () => {
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <QueryClientProvider client={queryClient}>
            <DndContext>
              <WorkflowFieldInput template={VIDEO_TEMPLATE} value={undefined} onChange={vi.fn()} />
            </DndContext>
          </QueryClientProvider>
        </ChakraProvider>
      );
    });

    expect(host.textContent).not.toContain('Connection only');
    expect(findButton('widgets.gallery.picker.chooseVideo').disabled).toBe(false);
  });

  it('renders a direct-input widget for VideoField instead of falling back to connection-only', async () => {
    await renderField(VIDEO_TEMPLATE, undefined, vi.fn());

    expect(host.textContent).not.toContain('Connection only');
    expect(findButton('widgets.gallery.picker.chooseVideo').disabled).toBe(false);
    expect(pickerState.accept).toEqual(['video']);
    expect(findButton('Upload').disabled).toBe(false);
    expect(host.querySelector<HTMLInputElement>('input[type="file"]')?.accept).toBe('video/*,audio/*');
  });

  it('adopts a video picked from the gallery, shows its details, and clears it', async () => {
    const onChange = vi.fn();

    await renderField(VIDEO_TEMPLATE, undefined, onChange);
    await act(() => findButton('widgets.gallery.picker.chooseVideo').click());
    expect(onChange).toHaveBeenCalledWith({ video_name: 'clip.mp4' });

    await renderField(VIDEO_TEMPLATE, { video_name: 'clip.mp4' }, onChange);
    expect(host.querySelector('img')?.src).toContain('/api/v1/videos/i/clip.mp4/thumbnail');
    await vi.waitFor(() => {
      // Dimensions/duration badge from the resolved item details.
      expect(host.textContent).toContain('640x480 · 5s');
    });

    await act(() => findButton('Clear').click());
    expect(onChange).toHaveBeenCalledWith(undefined);
  });

  it('edits image collections as a list: picks append, remove drops one, clear empties', async () => {
    const onChange = vi.fn();
    const collectionTemplate = {
      name: 'images',
      title: 'Images',
      type: { cardinality: 'COLLECTION', name: 'ImageField' },
    } as unknown as FieldInputTemplate;

    await renderField(collectionTemplate, [{ image_name: 'a.png' }, { image_name: 'b.png' }], onChange);

    expect(host.textContent).not.toContain('Connection only');
    expect(host.querySelectorAll('img')).toHaveLength(2);
    expect(host.querySelector<HTMLInputElement>('input[type="file"]')?.multiple).toBe(true);

    await act(() => findButton('common.add').click());
    expect(onChange).toHaveBeenLastCalledWith([
      { image_name: 'a.png' },
      { image_name: 'b.png' },
      { image_name: SELECTED_GALLERY_VIDEO.name },
    ]);

    await act(() => host.querySelector<HTMLButtonElement>('button[aria-label="Remove a.png"]')!.click());
    expect(onChange).toHaveBeenLastCalledWith([{ image_name: 'b.png' }]);

    await act(() => findButton('Clear').click());
    expect(onChange).toHaveBeenLastCalledWith([]);
  });

  it('keeps an empty image collection droppable and pickable', async () => {
    const onChange = vi.fn();
    const collectionTemplate = {
      name: 'images',
      title: 'Images',
      type: { cardinality: 'COLLECTION', name: 'ImageField' },
    } as unknown as FieldInputTemplate;

    await renderField(collectionTemplate, [], onChange);

    const emptyState = findButton('widgets.gallery.picker.chooseImage');

    expect(emptyState.getBoundingClientRect().height).toBeGreaterThanOrEqual(64);
    await act(() => emptyState.click());
    expect(onChange).toHaveBeenLastCalledWith([{ image_name: SELECTED_GALLERY_VIDEO.name }]);
  });

  it('adopts the files that uploaded when one of a multi-file batch fails', async () => {
    uploadImageMock.mockResolvedValueOnce({ imageName: 'first.png' }).mockRejectedValueOnce(new Error('413'));
    const createToast = vi.spyOn(toaster, 'create').mockImplementation(() => '');
    const onChange = vi.fn();
    const collectionTemplate = {
      name: 'images',
      title: 'Images',
      type: { cardinality: 'COLLECTION', name: 'ImageField' },
    } as unknown as FieldInputTemplate;

    await renderField(collectionTemplate, [{ image_name: 'a.png' }], onChange);

    const fileInput = host.querySelector<HTMLInputElement>('input[type="file"]')!;
    const transfer = new DataTransfer();

    transfer.items.add(new File(['1'], 'one.png', { type: 'image/png' }));
    transfer.items.add(new File(['2'], 'two.png', { type: 'image/png' }));
    fileInput.files = transfer.files;
    await act(() => fileInput.dispatchEvent(new Event('change', { bubbles: true })));

    await vi.waitFor(() => {
      expect(onChange).toHaveBeenCalledWith([{ image_name: 'a.png' }, { image_name: 'first.png' }]);
    });
    await vi.waitFor(() => {
      expect(createToast).toHaveBeenCalledWith(expect.objectContaining({ description: 'two.png', type: 'error' }));
    });
    createToast.mockRestore();
  });

  it('keeps video COLLECTION fields connection-only (the widget would write a bare object into a list)', async () => {
    const collectionTemplate = {
      name: 'videos',
      title: 'Videos',
      type: { cardinality: 'COLLECTION', name: 'VideoField' },
    } as unknown as FieldInputTemplate;

    await renderField(collectionTemplate, undefined, vi.fn());

    expect(host.textContent).toContain('Connection only');
    expect(host.querySelector('input[type="file"]')).toBeNull();
  });

  it('rejects a file whose type does not match the field kind', async () => {
    const onChange = vi.fn();

    await renderField(VIDEO_TEMPLATE, undefined, onChange);

    const fileInput = host.querySelector<HTMLInputElement>('input[type="file"]')!;
    const transfer = new DataTransfer();

    transfer.items.add(new File(['data'], 'a.png', { type: 'image/png' }));
    fileInput.files = transfer.files;
    await act(() => fileInput.dispatchEvent(new Event('change', { bubbles: true })));

    expect(uploadVideoMock).not.toHaveBeenCalled();
    expect(onChange).not.toHaveBeenCalled();
  });

  it('uploads a picked file and adopts the uploaded video', async () => {
    uploadVideoMock.mockResolvedValue({ kind: 'video', name: 'uploaded.mp4' });
    const onChange = vi.fn();

    await renderField(VIDEO_TEMPLATE, undefined, onChange);
    await act(() => findButton('Upload').click());

    const fileInput = host.querySelector<HTMLInputElement>('input[type="file"]')!;
    const transfer = new DataTransfer();

    transfer.items.add(new File(['data'], 'clip.mp4', { type: 'video/mp4' }));
    fileInput.files = transfer.files;
    await act(() => fileInput.dispatchEvent(new Event('change', { bubbles: true })));

    await vi.waitFor(() => {
      expect(onChange).toHaveBeenCalledWith({ video_name: 'uploaded.mp4' });
    });
    expect(uploadVideoMock).toHaveBeenCalledTimes(1);
    expect(uploadVideoMock.mock.calls[0]?.[1]).toBe('none');
  });

  it('renders the frame scrubber for video-frame-index integer fields', async () => {
    resolveItemMock.mockResolvedValue({ ...SELECTED_GALLERY_VIDEO, fps: 30 });
    graphNodes.push(makeFrameNode({ video_name: 'clip.mp4' }));

    // Default of -1 (= last frame) resolves against duration * fps = 150 frames.
    await renderField(FRAME_INDEX_TEMPLATE, -1, vi.fn(), 'frame-node');

    await vi.waitFor(() => {
      expect(host.querySelector('video')?.src).toContain('/api/v1/videos/i/clip.mp4/full');
    });
    expect(host.textContent).toContain('149 / 149');
    expect(host.querySelector('input[type="number"]')).not.toBeNull();
    expect(host.querySelector('[role="slider"]')).not.toBeNull();
  });

  it('scrubs frames with the slider, writing the resolved index to the field', async () => {
    resolveItemMock.mockResolvedValue({ ...SELECTED_GALLERY_VIDEO, fps: 30 });
    graphNodes.push(makeFrameNode({ video_name: 'clip.mp4' }));
    const onChange = vi.fn();

    await renderField(FRAME_INDEX_TEMPLATE, 10, onChange, 'frame-node');

    await vi.waitFor(() => {
      expect(host.querySelector('[role="slider"]')).not.toBeNull();
    });

    const thumb = host.querySelector<HTMLElement>('[role="slider"]')!;

    thumb.focus();
    await act(() => thumb.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' })));
    expect(onChange).toHaveBeenCalledWith(11);
  });

  it('falls back to a hint when the companion video field is unset', async () => {
    graphNodes.push(makeFrameNode(undefined));

    await renderField(FRAME_INDEX_TEMPLATE, 0, vi.fn(), 'frame-node');

    expect(host.querySelector('input[type="number"]')).not.toBeNull();
    expect(host.querySelector('video')).toBeNull();
    expect(host.textContent).toContain("Set this node's Video field to preview frames.");
  });

  it('falls back to a hint when the video has no probed frame rate', async () => {
    // SELECTED_GALLERY_VIDEO has no fps, so frames cannot be mapped onto time.
    graphNodes.push(makeFrameNode({ video_name: 'clip.mp4' }));

    await renderField(FRAME_INDEX_TEMPLATE, 0, vi.fn(), 'frame-node');

    await vi.waitFor(() => {
      expect(host.textContent).toContain('no probed frame rate');
    });
    expect(host.querySelector('video')).toBeNull();
  });

  it('ignores the stored video value while the video field is connection-driven', async () => {
    resolveItemMock.mockResolvedValue({ ...SELECTED_GALLERY_VIDEO, fps: 30 });
    graphNodes.push(makeFrameNode({ video_name: 'clip.mp4' }), {
      data: { inputs: {}, type: 'video' },
      id: 'upstream',
      type: 'invocation',
    });
    graphEdges.push({
      id: 'e1',
      source: 'upstream',
      sourceHandle: 'video',
      target: 'frame-node',
      targetHandle: 'video',
    });

    await renderField(FRAME_INDEX_TEMPLATE, 0, vi.fn(), 'frame-node');

    expect(host.textContent).toContain('comes from a graph connection');
    expect(host.querySelector('video')).toBeNull();
    expect(resolveItemMock).not.toHaveBeenCalled();
  });

  it('shows the preview without a slider for a single-frame video', async () => {
    // min === max would render a broken (NaN%) zag slider.
    resolveItemMock.mockResolvedValue({ ...SELECTED_GALLERY_VIDEO, durationSeconds: 0.02, fps: 30 });
    graphNodes.push(makeFrameNode({ video_name: 'clip.mp4' }));

    await renderField(FRAME_INDEX_TEMPLATE, 0, vi.fn(), 'frame-node');

    await vi.waitFor(() => {
      expect(host.querySelector('video')).not.toBeNull();
    });
    expect(host.textContent).toContain('0 / 0');
    expect(host.querySelector('[role="slider"]')).toBeNull();
  });

  it('reports a deleted video instead of a frame-rate story', async () => {
    resolveItemMock.mockRejectedValue(new Error('404'));
    graphNodes.push(makeFrameNode({ video_name: 'gone.mp4' }));

    await renderField(FRAME_INDEX_TEMPLATE, 0, vi.fn(), 'frame-node');

    await vi.waitFor(() => {
      expect(host.textContent).toContain('could not be loaded');
    });
    expect(host.querySelector('video')).toBeNull();
  });

  it('treats an empty-string video name as unset without firing a lookup', async () => {
    graphNodes.push(makeFrameNode({ video_name: '' }));

    await renderField(FRAME_INDEX_TEMPLATE, 0, vi.fn(), 'frame-node');

    expect(host.textContent).toContain("Set this node's Video field to preview frames.");
    expect(resolveItemMock).not.toHaveBeenCalled();
  });

  it('keeps the image field on the image upload path', async () => {
    uploadImageMock.mockResolvedValue({ imageName: 'uploaded.png' });
    const onChange = vi.fn();
    const imageTemplate = {
      name: 'image',
      title: 'Image',
      type: { name: 'ImageField' },
    } as unknown as FieldInputTemplate;

    await renderField(imageTemplate, undefined, onChange);

    expect(host.querySelector<HTMLInputElement>('input[type="file"]')?.accept).toBe('image/*');
    await act(() => findButton('Upload').click());

    const fileInput = host.querySelector<HTMLInputElement>('input[type="file"]')!;
    const transfer = new DataTransfer();

    transfer.items.add(new File(['data'], 'a.png', { type: 'image/png' }));
    fileInput.files = transfer.files;
    await act(() => fileInput.dispatchEvent(new Event('change', { bubbles: true })));

    await vi.waitFor(() => {
      expect(onChange).toHaveBeenCalledWith({ image_name: 'uploaded.png' });
    });
    expect(uploadVideoMock).not.toHaveBeenCalled();
  });
});

/** Feed committed values back through an owning parent to expose keystroke rerenders and blur clamping. */
const StatefulLoRAField = ({ initial, onCommit }: { initial: unknown; onCommit: (value: unknown) => void }) => {
  const [value, setValue] = useState(initial);
  const onChange = useCallback(
    (next: unknown) => {
      setValue(next);
      onCommit(next);
    },
    [onCommit]
  );

  return <WorkflowFieldInput template={LORA_COLLECTION_TEMPLATE} value={value} onChange={onChange} />;
};

describe('WorkflowFieldInput LoRA collection', () => {
  const renderStatefulLoras = async (initial: unknown, onCommit: (value: unknown) => void) => {
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <QueryClientProvider client={queryClient}>
            <DndContext>
              <StatefulLoRAField initial={initial} onCommit={onCommit} />
            </DndContext>
          </QueryClientProvider>
        </ChakraProvider>
      );
    });
    await vi.waitFor(
      () => {
        expect(modelSelectState.props).not.toBeNull();
      },
      { timeout: 5_000 }
    );
  };

  const renderLoras = async (value: unknown, onChange: (value: unknown) => void) => {
    await renderField(LORA_COLLECTION_TEMPLATE, value, onChange);
    // The picker is lazy; wait for Suspense to resolve it before asserting on the mounted widget.
    await vi.waitFor(
      () => {
        expect(modelSelectState.props).not.toBeNull();
      },
      { timeout: 5_000 }
    );
  };

  it('scopes the picker to the LoRAs the node can apply and hides the ones already added', async () => {
    await renderLoras([loraEntry()], vi.fn());

    const props = modelSelectState.props!;

    expect(props.modelTypes).toEqual(['lora']);
    expect(Array.from(props.excludeKeys ?? [])).toEqual(['sd-lora']);
    expect(props.filter?.(SD_LORA)).toBe(true);
    expect(props.filter?.(SDXL_LORA)).toBe(false);
  });

  it("appends a picked LoRA at the model's own default weight", async () => {
    const onChange = vi.fn();

    await renderLoras([loraEntry()], onChange);
    await act(() => findButton('Add LoRA…').click());

    // 0.4 comes from ADDABLE_LORA.default_settings, not from DEFAULT_LORA_WEIGHT_CONFIG.initial.
    expect(onChange).toHaveBeenCalledWith([
      loraEntry(),
      { lora: { base: 'sd-1', hash: 'hash-add', key: 'add-lora', name: 'Add Me', type: 'lora' }, weight: 0.4 },
    ]);
  });

  it('edits one weight and removes one entry without disturbing the rest', async () => {
    const onChange = vi.fn();
    const other = loraEntry({
      lora: { base: 'sd-1', hash: 'hash-other', key: 'other-lora', name: 'Other', type: 'lora' },
      weight: 1,
    });

    await renderLoras([loraEntry(), other], onChange);

    const weightInputs = Array.from(host.querySelectorAll<HTMLInputElement>('input'));

    expect(weightInputs).toHaveLength(2);
    expect(host.textContent).toContain('Detail Tweaker');
    expect(host.textContent).toContain('Other');

    await typeWeight(weightInputs[0]!, '0.5');
    expect(onChange).toHaveBeenCalledWith([loraEntry({ weight: 0.5 }), other]);

    const removeOther = host.querySelector<HTMLButtonElement>('button[aria-label="Remove Other"]')!;

    await act(() => removeOther.click());
    expect(onChange).toHaveBeenLastCalledWith([loraEntry()]);
  });

  it('edits the right row when the same LoRA appears twice in an imported workflow', async () => {
    const onChange = vi.fn();

    await renderLoras([loraEntry(), loraEntry({ weight: 1 })], onChange);

    const removeButtons = host.querySelectorAll<HTMLButtonElement>('button[aria-label="Remove Detail Tweaker"]');

    expect(removeButtons).toHaveLength(2);
    await act(() => removeButtons[1]!.click());
    expect(onChange).toHaveBeenCalledWith([loraEntry()]);
  });

  it('keeps a half-typed decimal instead of snapping back to the last committed number', async () => {
    const onChange = vi.fn();

    await renderStatefulLoras([loraEntry()], onChange);

    const weight = host.querySelector<HTMLInputElement>('input')!;

    // Test each keystroke so trailing-decimal drafts cannot reset to zero and turn 0.5 into 5.
    await typeWeight(weight, '0');
    await pressKey('.');

    // A number input reports the trailing `.` as absent; the next digit landing after it is the proof.
    expect(weight.value).toBe('0');
    expect(onChange).toHaveBeenLastCalledWith([loraEntry({ weight: 0 })]);

    await pressKey('5');

    expect(weight.value).toBe('0.5');
    expect(onChange).toHaveBeenLastCalledWith([loraEntry({ weight: 0.5 })]);
    // Nothing on the way there was ever an order of magnitude out.
    expect(onChange.mock.calls.every(([value]) => (value as { weight: number }[])[0]!.weight <= 0.5)).toBe(true);

    // The arrow keys step by the LoRA coarse step, not the number input's default of 1.
    await pressKey('{ArrowUp}');
    expect(weight.value).toBe('0.55');
    expect(onChange).toHaveBeenLastCalledWith([loraEntry({ weight: 0.55 })]);
  });

  it('keeps an out-of-range or cleared weight on screen and invalid instead of clamping or restoring it', async () => {
    const onChange = vi.fn();

    await renderStatefulLoras([loraEntry()], onChange);

    const weight = host.querySelector<HTMLInputElement>('input')!;

    await typeWeight(weight, '999');
    expect(onChange).toHaveBeenLastCalledWith([loraEntry({ weight: 999 })]);
    expect(weight.getAttribute('aria-invalid')).toBe('true');
    await blurWeight(weight);
    expect(weight.value).toBe('999');
    expect(weight.getAttribute('aria-invalid')).toBe('true');

    await typeWeight(weight, '2');
    expect(onChange).toHaveBeenLastCalledWith([loraEntry({ weight: 2 })]);
    expect(weight.getAttribute('aria-invalid')).toBeNull();

    // Clearing keeps the row (and its input) and reports through the field, not by snapping back to 2.
    await typeWeight(weight, '{Backspace}');
    expect(onChange).toHaveBeenLastCalledWith([loraEntry({ weight: null })]);
    expect(weight.value).toBe('');
    expect(weight.getAttribute('aria-invalid')).toBe('true');
    await blurWeight(weight);
    expect(host.querySelector<HTMLInputElement>('input')).toBe(weight);
    expect(weight.value).toBe('');
    expect(weight.getAttribute('aria-invalid')).toBe('true');
  });

  it('keeps an unreadable entry in the list and lets the user remove it', async () => {
    const onChange = vi.fn();
    // A key-only identifier is not renderable and the backend would reject it.
    const broken = { lora: { key: 'ghost' }, weight: 1 };

    await renderLoras([loraEntry(), broken], onChange);

    expect(host.textContent).toContain('Unreadable entry');
    // Only the readable row gets a weight control.
    expect(host.querySelectorAll('input')).toHaveLength(1);

    // Editing the good row must not drop the bad one.
    await typeWeight(host.querySelector<HTMLInputElement>('input')!, '0.5');
    expect(onChange).toHaveBeenLastCalledWith([loraEntry({ weight: 0.5 }), broken]);

    await act(() => host.querySelector<HTMLButtonElement>('button[aria-label="Remove Unreadable entry"]')!.click());
    expect(onChange).toHaveBeenLastCalledWith([loraEntry()]);
  });

  it('clears the field back to its default when the last entry is removed', async () => {
    const onChange = vi.fn();

    await renderLoras([loraEntry()], onChange);
    await act(() => host.querySelector<HTMLButtonElement>('button[aria-label="Remove Detail Tweaker"]')!.click());

    // `undefined`, not `[]`: the loaders default to no LoRAs, so the node returns to its default.
    expect(onChange).toHaveBeenCalledWith(undefined);
  });

  it('renders a single stored LoRAField as a one-row list rather than falling back', async () => {
    await renderLoras(loraEntry(), vi.fn());

    expect(host.textContent).not.toContain('Connection only');
    expect(host.querySelectorAll('input')).toHaveLength(1);
  });
});

describe('WorkflowFieldInput model identifiers', () => {
  const ltx2Template = (uiModelFormat: string[] | null) =>
    ({
      name: 'model',
      title: 'Model',
      type: { batch: false, cardinality: 'SINGLE', name: 'ModelIdentifierField' },
      uiModelBase: ['ltx-2'],
      uiModelFormat,
      uiModelType: ['main'],
    }) as unknown as FieldInputTemplate;
  const transformer = { base: 'ltx-2', format: 'checkpoint', key: 'dev', type: 'main' };
  const fullFolder = { base: 'ltx-2', components_only: false, format: 'diffusers', key: 'full', type: 'main' };
  const componentsFolder = { base: 'ltx-2', components_only: true, format: 'diffusers', key: 'parts', type: 'main' };

  const pickerFilter = async (template: FieldInputTemplate) => {
    await renderField(template, undefined, vi.fn());
    await vi.waitFor(
      () => {
        expect(modelSelectState.props).not.toBeNull();
      },
      { timeout: 5_000 }
    );
    return modelSelectState.props!.filter!;
  };

  it('keeps components-only folders out of a format-agnostic model field', async () => {
    const filter = await pickerFilter(ltx2Template(null));

    expect(filter(transformer)).toBe(true);
    expect(filter(fullFolder)).toBe(true);
    expect(filter(componentsFolder)).toBe(false);
    expect(filter({ ...transformer, base: 'wan' })).toBe(false);
  });

  it('offers components-only folders to a field that asks for folders', async () => {
    const filter = await pickerFilter(ltx2Template(['diffusers']));

    expect(filter(componentsFolder)).toBe(true);
    expect(filter(transformer)).toBe(false);
  });
});

describe('WorkflowFieldInput seed inputs', () => {
  const settle = async (action: () => void) => {
    await act(async () => {
      action();
      await new Promise<void>((resolve) => {
        setTimeout(resolve, 0);
      });
    });
  };
  const seedInput = () => host.querySelector<HTMLInputElement>('input[aria-label="Seed"]');
  const diceButton = () => host.querySelector<HTMLButtonElement>('button[aria-label="common.newSeed"]');
  const modeTrigger = () => host.querySelector<HTMLButtonElement>('button[aria-haspopup="menu"]');
  const menuItem = (mode: string) =>
    document.querySelector<HTMLElement>(`[role="menuitemradio"][data-value="${mode}"]`);

  it('renders a plain number input when no seed mode is supplied, and for non-seed integers', async () => {
    await renderField(SEED_TEMPLATE, 42, vi.fn());

    expect(seedInput()?.value).toBe('42');
    expect(modeTrigger()).toBeNull();
    expect(diceButton()).toBeNull();

    await renderField({ ...SEED_TEMPLATE, name: 'steps' } as FieldInputTemplate, 20, vi.fn(), undefined, {
      onSeedModeChange: vi.fn(),
      seedMode: 'increment',
    });

    expect(modeTrigger()).toBeNull();
  });

  it('puts the mode menu beside a seed input and commits the chosen mode without touching the value', async () => {
    const onChange = vi.fn();
    const onSeedModeChange = vi.fn();

    // Box the row at the node width so the width assertions below mean what they say; the
    // largest seed has to fit beside the stepper, the dice and the trigger without scrolling.
    host.style.width = '18rem';
    await renderField(SEED_TEMPLATE, 4_294_967_295, onChange, undefined, { onSeedModeChange, seedMode: 'fixed' });

    expect(seedInput()?.disabled).toBe(false);
    // Use untranslated long keys to verify truncation preserves accessible names and number-input width.
    expect(modeTrigger()?.getAttribute('aria-label')).toBe('common.seedMode.label: common.seedMode.fixed');
    expect(modeTrigger()?.getBoundingClientRect().width).toBeLessThanOrEqual(144);
    expect((seedInput() as HTMLInputElement).scrollWidth).toBeLessThanOrEqual(
      (seedInput() as HTMLInputElement).clientWidth
    );
    // xyflow reads `.nokey` to leave a control's keys alone; the trigger and the portaled menu both need it,
    // or arrows nudge the node and Backspace deletes it while the menu is in use.
    expect(modeTrigger()?.closest('.nokey')).not.toBeNull();

    await settle(() => modeTrigger()?.click());

    expect(menuItem('fixed')?.getAttribute('aria-checked')).toBe('true');
    expect(menuItem('fixed')?.closest('.nokey')).not.toBeNull();

    await settle(() => menuItem('increment')?.click());

    expect(onSeedModeChange).toHaveBeenCalledWith('increment');
    expect(onChange).not.toHaveBeenCalled();
  });

  it("previews the next batch from the workflow's own run count and describes the input with it", async () => {
    workflowValues.batchCount = 3;

    await renderField(SEED_TEMPLATE, 42, vi.fn(), undefined, { onSeedModeChange: vi.fn(), seedMode: 'increment' });

    const preview = host.querySelector<HTMLElement>('[data-testid="seed-sequence-preview"]');

    // Without i18n resources the key renders, carrying the interpolated bounds.
    expect(preview?.textContent).toBe('common.seedNextBatchRange');
    expect(seedInput()?.getAttribute('aria-describedby')).toBe(preview?.id);

    await renderField(SEED_TEMPLATE, 42, vi.fn(), undefined, { onSeedModeChange: vi.fn(), seedMode: 'fixed' });

    expect(host.querySelector('[data-testid="seed-sequence-preview"]')).toBeNull();
    expect(seedInput()?.getAttribute('aria-describedby')).toBeNull();

    // An empty field previews from the template default, which is where the plan starts it.
    workflowValues.batchCount = 2;
    await renderField({ ...SEED_TEMPLATE, default: 1_234 } as FieldInputTemplate, undefined, vi.fn(), undefined, {
      onSeedModeChange: vi.fn(),
      seedMode: 'increment',
    });

    expect(host.querySelector('[data-testid="seed-sequence-preview"]')?.textContent).toBe('common.seedNextBatchRange');
  });

  it('rolls a new seed from the dice and writes it to the field as an integer in range', async () => {
    const onChange = vi.fn();

    await renderField(SEED_TEMPLATE, 42, onChange, undefined, { onSeedModeChange: vi.fn(), seedMode: 'fixed' });
    await settle(() => diceButton()?.click());

    expect(onChange).toHaveBeenCalledTimes(1);
    const rolled = onChange.mock.calls[0]?.[0] as number;

    expect(Number.isInteger(rolled)).toBe(true);
    expect(rolled).toBeGreaterThanOrEqual(0);
    expect(rolled).toBeLessThanOrEqual(4_294_967_295);
  });

  it('quiets the value and the dice in random mode but keeps the seed on show', async () => {
    await renderField(SEED_TEMPLATE, 42, vi.fn(), undefined, { onSeedModeChange: vi.fn(), seedMode: 'random' });

    expect(seedInput()?.disabled).toBe(true);
    expect(seedInput()?.value).toBe('42');
    expect(diceButton()?.disabled).toBe(true);
  });
});

const fullTemplate = (
  typeName: 'FloatField' | 'IntegerField' | 'StringField',
  overrides: Partial<FieldInputTemplate> = {}
): FieldInputTemplate => ({
  default: undefined,
  description: '',
  exclusiveMaximum: null,
  exclusiveMinimum: null,
  fieldKind: 'input',
  input: 'any',
  maximum: null,
  minimum: null,
  multipleOf: null,
  name: 'value',
  options: null,
  required: true,
  title: 'Value',
  type: { batch: false, cardinality: 'SINGLE', name: typeName },
  uiChoiceLabels: null,
  uiComponent: null,
  uiHidden: false,
  uiModelBase: null,
  uiModelFormat: null,
  uiModelType: null,
  uiOrder: null,
  ...overrides,
});

/**
 * Feeds each commit back as the next prop and derives `invalid` the way both hosts do, so echo-driven caret bugs
 * reproduce. A deferred echo lands in a transition, as the Canvas flow-model rebuild does.
 */
const StatefulField = ({
  echo = 'sync',
  initial,
  onCommit,
  template,
}: {
  echo?: 'deferred' | 'sync';
  initial: unknown;
  onCommit: (value: unknown) => void;
  template: FieldInputTemplate;
}) => {
  const [value, setValue] = useState(initial);
  const onChange = useCallback(
    (next: unknown) => {
      if (echo === 'deferred') {
        startTransition(() => setValue(next));
      } else {
        setValue(next);
      }
      onCommit(next);
    },
    [echo, onCommit]
  );
  const invalid = getWorkflowFieldInvalidReason({ isConnected: false, template, value }) !== null;

  // Both editors wrap the control in a Field.Root that marks everything inside it invalid.
  return (
    <Field.Root invalid={invalid}>
      <WorkflowFieldInput invalid={invalid} template={template} value={value} onChange={onChange} />
    </Field.Root>
  );
};

describe('WorkflowFieldInput text and number entry', () => {
  let renderCount = 0;
  const renderStateful = async (template: FieldInputTemplate, initial: unknown, echo?: 'deferred' | 'sync') => {
    const onCommit = vi.fn();

    // Keyed so a later render in the same test starts from its own initial value.
    renderCount += 1;
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <StatefulField key={renderCount} echo={echo} initial={initial} template={template} onCommit={onCommit} />
        </ChakraProvider>
      );
    });

    const input = host.querySelector<HTMLInputElement | HTMLTextAreaElement>('input, textarea');

    if (!input) {
      throw new Error('Field input not rendered');
    }

    // The click lands past the end of a short value; End makes the starting caret explicit either way.
    await act(async () => {
      await userEvent.click(input);
      await userEvent.keyboard('{End}');
    });

    return { input, onCommit };
  };
  const keys = (sequence: string) =>
    act(async () => {
      await userEvent.keyboard(sequence);
    });
  const blur = (input: HTMLElement) =>
    act(() => {
      input.blur();
    });
  const caret = (input: HTMLInputElement | HTMLTextAreaElement) => [input.selectionStart, input.selectionEnd];
  const isInvalid = (input: HTMLElement) => input.getAttribute('aria-invalid') === 'true';
  const FLOAT = fullTemplate('FloatField');
  const INTEGER = fullTemplate('IntegerField');
  const TEXT = fullTemplate('StringField');

  describe('text', () => {
    it('keeps the caret at the edit point while each keystroke echoes back through the value', async () => {
      const { input, onCommit } = await renderStateful(TEXT, 'hello world');

      await keys('{Home}X');
      expect(input.value).toBe('Xhello world');
      expect(caret(input)).toEqual([1, 1]);

      await keys('{End}{ArrowLeft}{ArrowLeft}{ArrowLeft}{ArrowLeft}{ArrowLeft}Y');
      expect(input.value).toBe('Xhello Yworld');
      expect(caret(input)).toEqual([8, 8]);

      await keys('{End}Z');
      expect(input.value).toBe('Xhello YworldZ');
      expect(caret(input)).toEqual([14, 14]);
      expect(onCommit).toHaveBeenLastCalledWith('Xhello YworldZ');
      expect(document.activeElement).toBe(input);
    });

    it('keeps every keystroke and the caret when the committed value echoes back late', async () => {
      const { input, onCommit } = await renderStateful(TEXT, 'hello world', 'deferred');

      // Two keystrokes inside one act: the second lands before the first commit has echoed back.
      await keys('{Home}{ArrowRight}ab');
      expect(input.value).toBe('habello world');
      expect(caret(input)).toEqual([3, 3]);
      expect(onCommit).toHaveBeenLastCalledWith('habello world');

      const number = await renderStateful(FLOAT, 0.5, 'deferred');

      await keys('{Home}12');
      expect(number.input.value).toBe('120.5');
      expect(number.onCommit).toHaveBeenLastCalledWith(120.5);
      await keys('3');
      expect(number.input.value).toBe('1230.5');
    });

    it('replaces a selection in place and leaves the caret after the replacement', async () => {
      const { input, onCommit } = await renderStateful(TEXT, 'hello world');

      await keys('{Shift>}{ArrowLeft}{ArrowLeft}{/Shift}ab');
      expect(input.value).toBe('hello worab');
      expect(caret(input)).toEqual([11, 11]);

      await keys('{Control>}{Shift>}{ArrowLeft}{/Shift}{/Control}there');
      expect(input.value).toBe('hello there');
      expect(caret(input)).toEqual([11, 11]);

      await keys('{Home}{ArrowRight}{ArrowRight}{Shift>}{ArrowRight}{/Shift}L');
      expect(input.value).toBe('heLlo there');
      expect(caret(input)).toEqual([3, 3]);

      await keys('{Control>}a{/Control}new value');
      expect(input.value).toBe('new value');
      expect(caret(input)).toEqual([9, 9]);
      expect(onCommit).toHaveBeenLastCalledWith('new value');

      await act(async () => {
        await userEvent.dblClick(input);
      });
      await keys('word');
      expect(input.value).toBe('new word');
      expect(caret(input)).toEqual([8, 8]);
    });

    it('deletes at the caret, holds still at the boundaries, and stays empty once cleared', async () => {
      const { input, onCommit } = await renderStateful(TEXT, 'abc');

      await keys('{Home}{Backspace}');
      expect(input.value).toBe('abc');
      expect(caret(input)).toEqual([0, 0]);

      await keys('{End}{Delete}');
      expect(input.value).toBe('abc');
      expect(caret(input)).toEqual([3, 3]);

      await keys('{ArrowLeft}{Backspace}');
      expect(input.value).toBe('ac');
      expect(caret(input)).toEqual([1, 1]);

      await keys('{Home}{Delete}');
      expect(input.value).toBe('c');
      expect(caret(input)).toEqual([0, 0]);

      await keys('{Delete}');
      expect(input.value).toBe('');
      expect(onCommit).toHaveBeenLastCalledWith('');
      expect(document.activeElement).toBe(input);

      await keys('xyz{Control>}a{/Control}{Backspace}');
      expect(input.value).toBe('');
      expect(onCommit).toHaveBeenLastCalledWith('');
      expect(document.activeElement).toBe(input);
    });

    it('pastes at the caret and over a selection, and keeps spacing and punctuation as typed', async () => {
      const { input, onCommit } = await renderStateful(TEXT, 'hello world');

      await keys('{Control>}{Shift>}{ArrowLeft}{/Shift}{/Control}');
      await act(async () => {
        await userEvent.copy();
      });
      await keys('{Home}');
      await act(async () => {
        await userEvent.paste();
      });
      expect(input.value).toBe('worldhello world');
      expect(caret(input)).toEqual([5, 5]);

      await keys('{End}{ArrowLeft}{ArrowLeft}{ArrowLeft}');
      await act(async () => {
        await userEvent.paste();
      });
      expect(input.value).toBe('worldhello woworldrld');
      expect(caret(input)).toEqual([18, 18]);

      await keys('{Control>}a{/Control}');
      await act(async () => {
        await userEvent.paste();
      });
      expect(input.value).toBe('world');
      expect(caret(input)).toEqual([5, 5]);

      await keys(', again.  ');
      expect(input.value).toBe('world, again.  ');
      expect(onCommit).toHaveBeenLastCalledWith('world, again.  ');
    });

    it('keeps a composed entry intact while each update echoes back, in the input and the textarea', async () => {
      const TEXTAREA = fullTemplate('StringField', { uiComponent: 'textarea' });

      for (const [template, echo] of [
        [TEXT, 'sync'],
        [TEXT, 'deferred'],
        [TEXTAREA, 'deferred'],
      ] as const) {
        const { input, onCommit } = await renderStateful(template, 'hello world', echo);

        await keys('{Home}{ArrowRight}{ArrowRight}');
        // Kana-style entry: every update replaces the candidate text; the commit inserts the converted form.
        await act(async () => {
          await server.commands.imeCompose([
            { kind: 'compose', text: 'n' },
            { kind: 'compose', text: 'に' },
            { kind: 'compose', text: 'にh' },
            { kind: 'compose', text: 'にほ' },
            { kind: 'compose', text: 'にほん' },
          ]);
        });
        expect(input.value).toBe('heにほんllo world');
        expect(caret(input)).toEqual([5, 5]);

        await act(async () => {
          await server.commands.imeCompose([{ kind: 'commit', text: '日本' }]);
        });
        expect(input.value).toBe('he日本llo world');
        expect(caret(input)).toEqual([4, 4]);
        expect(onCommit).toHaveBeenLastCalledWith('he日本llo world');

        await keys('!');
        expect(input.value).toBe('he日本!llo world');
        expect(onCommit).toHaveBeenLastCalledWith('he日本!llo world');
      }
    });

    it('keeps newlines and cross-line replacements in a textarea', async () => {
      const { input, onCommit } = await renderStateful(
        fullTemplate('StringField', { uiComponent: 'textarea' }),
        'alpha beta'
      );

      expect(input.tagName).toBe('TEXTAREA');

      await keys('{Home}{ArrowRight}{ArrowRight}{ArrowRight}{ArrowRight}{ArrowRight}{Enter}');
      expect(input.value).toBe('alpha\n beta');
      expect(caret(input)).toEqual([6, 6]);

      await keys('X');
      expect(input.value).toBe('alpha\nX beta');
      expect(caret(input)).toEqual([7, 7]);

      await keys('{Control>}{Home}{/Control}{ArrowRight}{ArrowRight}{ArrowRight}{Shift>}{ArrowDown}{/Shift}Q');
      expect(input.value).toBe('alpQeta');
      expect(caret(input)).toEqual([4, 4]);
      expect(onCommit).toHaveBeenLastCalledWith('alpQeta');

      await keys('{End}{Backspace}{Backspace}{Backspace}{Backspace}{Backspace}{Backspace}{Backspace}');
      expect(input.value).toBe('');
      expect(onCommit).toHaveBeenLastCalledWith('');
      expect(document.activeElement).toBe(input);
    });
  });

  describe('float', () => {
    it('inserts digits at the caret of a decimal and commits each result as typed', async () => {
      const cases: [string, string, string, number][] = [
        ['{Home}2', '20.5', '230.5', 230.5],
        ['{Home}{ArrowRight}2', '02.5', '023.5', 23.5],
        ['{End}{ArrowLeft}6', '0.65', '0.635', 0.635],
        ['{End}2', '0.52', '0.523', 0.523],
      ];

      for (const [sequence, afterInsert, afterNext, committed] of cases) {
        const { input, onCommit } = await renderStateful(FLOAT, 0.5);

        await keys(sequence);
        expect(input.value).toBe(afterInsert);
        // A number input hides its caret; where the next digit lands shows it stayed after the insertion.
        await keys('3');
        expect(input.value).toBe(afterNext);
        expect(onCommit).toHaveBeenLastCalledWith(committed);
        expect(document.activeElement).toBe(input);
      }
    });

    it('replaces a selected digit or the whole value without jumping after the first character', async () => {
      const { input, onCommit } = await renderStateful(FLOAT, 0.5);

      await keys('{Shift>}{ArrowLeft}{/Shift}6');
      expect(input.value).toBe('0.6');
      expect(onCommit).toHaveBeenLastCalledWith(0.6);

      await keys('7');
      expect(input.value).toBe('0.67');

      await keys('{Control>}a{/Control}0.6');
      expect(input.value).toBe('0.6');
      expect(onCommit).toHaveBeenLastCalledWith(0.6);

      await keys('{Control>}a{/Control}25');
      expect(input.value).toBe('25');
      expect(onCommit).toHaveBeenLastCalledWith(25);
      expect(document.activeElement).toBe(input);

      await act(async () => {
        await userEvent.dblClick(input);
      });
      await keys('4');
      expect(input.value).toBe('4');
      expect(onCommit).toHaveBeenLastCalledWith(4);
    });

    it('keeps decimal and sign drafts while they are incomplete', async () => {
      const fromZero = await renderStateful(FLOAT, 0);
      const commitsBeforeDot = fromZero.onCommit.mock.calls.length;

      await keys('.');
      expect(fromZero.onCommit.mock.calls.length).toBe(commitsBeforeDot);
      await keys('6');
      expect(fromZero.input.value).toBe('0.6');
      expect(fromZero.onCommit).toHaveBeenLastCalledWith(0.6);

      const fromEmpty = await renderStateful(FLOAT, undefined);

      await keys('.6');
      expect(fromEmpty.input.value).toBe('.6');
      expect(fromEmpty.onCommit).toHaveBeenLastCalledWith(0.6);
      await blur(fromEmpty.input);
      expect(fromEmpty.input.value).toBe('0.6');

      const signed = await renderStateful(FLOAT, undefined);

      await keys('-');
      expect(signed.input.value).toBe('');
      expect((signed.input as HTMLInputElement).validity.badInput).toBe(true);
      await keys('4');
      expect(signed.input.value).toBe('-4');
      expect(signed.onCommit).toHaveBeenLastCalledWith(-4);
      expect(isInvalid(signed.input)).toBe(false);

      await keys('{Control>}a{/Control}0.5{Home}-');
      expect(signed.input.value).toBe('-0.5');
      expect(signed.onCommit).toHaveBeenLastCalledWith(-0.5);
      await keys('{End}{Backspace}{Backspace}{Backspace}{Backspace}1.');
      await keys('5');
      expect(signed.input.value).toBe('1.5');
      expect(signed.onCommit.mock.calls.slice(-2).map(([value]) => value)).toEqual([1, 1.5]);

      await keys('{Control>}a{/Control}1e-3');
      expect(signed.input.value).toBe('1e-3');
      expect(signed.onCommit).toHaveBeenLastCalledWith(0.001);
      await blur(signed.input);
      expect(signed.input.value).toBe('0.001');
    });

    it('edits the decimal point in place and holds still at the boundaries', async () => {
      const { input, onCommit } = await renderStateful(FLOAT, 0.5);

      await keys('{Home}{ArrowRight}{Delete}');
      expect(input.value).toBe('05');
      expect(onCommit).toHaveBeenLastCalledWith(5);
      await keys('9');
      expect(input.value).toBe('095');

      await keys('{Control>}a{/Control}0.5{End}{ArrowLeft}{Backspace}');
      expect(input.value).toBe('05');
      await keys('9');
      expect(input.value).toBe('095');

      await keys('{Control>}a{/Control}45{Home}{Backspace}');
      expect(input.value).toBe('45');
      await keys('1');
      expect(input.value).toBe('145');

      await keys('{End}{Delete}');
      expect(input.value).toBe('145');
      await keys('2');
      expect(input.value).toBe('1452');
      expect(onCommit).toHaveBeenLastCalledWith(1452);
    });

    it('clears with Backspace, Delete, or a selection and stays empty and focused', async () => {
      const backspaced = await renderStateful(FLOAT, 4);

      await keys('{Backspace}');
      expect(backspaced.input.value).toBe('');
      expect(backspaced.onCommit).toHaveBeenLastCalledWith(undefined);
      expect(isInvalid(backspaced.input)).toBe(true);
      expect(document.activeElement).toBe(backspaced.input);
      await keys('7');
      expect(backspaced.input.value).toBe('7');
      expect(isInvalid(backspaced.input)).toBe(false);

      const deleted = await renderStateful(FLOAT, 4);

      await keys('{Home}{Delete}');
      expect(deleted.input.value).toBe('');
      expect(deleted.onCommit).toHaveBeenLastCalledWith(undefined);
      await keys('8');
      expect(deleted.input.value).toBe('8');

      const selected = await renderStateful(FLOAT, 1234);

      await keys('{Control>}a{/Control}{Delete}');
      expect(selected.input.value).toBe('');
      await keys('12{Control>}a{/Control}{Backspace}');
      expect(selected.input.value).toBe('');
      expect(selected.onCommit).toHaveBeenLastCalledWith(undefined);
      expect(document.activeElement).toBe(selected.input);

      const optional = await renderStateful(fullTemplate('FloatField', { required: false }), 4);

      await keys('{Backspace}');
      expect(optional.input.value).toBe('');
      expect(optional.onCommit).toHaveBeenLastCalledWith(undefined);
      expect(isInvalid(optional.input)).toBe(false);
    });

    it('steps with the arrow keys by the declared step and ignores the wheel', async () => {
      const stepped = await renderStateful(fullTemplate('FloatField', { multipleOf: 0.25 }), 0.5);

      await keys('{ArrowUp}');
      expect(stepped.input.value).toBe('0.75');
      expect(stepped.onCommit).toHaveBeenLastCalledWith(0.75);
      await keys('{ArrowDown}{ArrowDown}');
      expect(stepped.input.value).toBe('0.25');
      expect(stepped.onCommit).toHaveBeenLastCalledWith(0.25);
      expect(document.activeElement).toBe(stepped.input);

      // Chromium would step the focused value on wheel; the field lets the event through unfocused, then takes
      // focus back with the draft intact.
      await keys('{Control>}a{/Control}0.');
      expect(stepped.onCommit).toHaveBeenLastCalledWith(0);
      await act(async () => {
        await userEvent.wheel(stepped.input, { delta: { y: 100 } });
      });
      await vi.waitFor(() => {
        expect(document.activeElement).toBe(stepped.input);
      });
      expect(stepped.onCommit).toHaveBeenLastCalledWith(0);
      await keys('5');
      expect(stepped.input.value).toBe('0.5');
      expect(stepped.onCommit).toHaveBeenLastCalledWith(0.5);

      const integer = await renderStateful(INTEGER, 3);

      await keys('{ArrowUp}');
      expect(integer.input.value).toBe('4');
      expect(integer.onCommit).toHaveBeenLastCalledWith(4);
    });

    it('pastes a number at the caret and over a selection', async () => {
      const { input, onCommit } = await renderStateful(FLOAT, 25);

      await keys('{Home}{Shift>}{ArrowRight}{/Shift}');
      await act(async () => {
        await userEvent.copy();
      });
      await keys('{End}');
      await act(async () => {
        await userEvent.paste();
      });
      expect(input.value).toBe('252');
      expect(onCommit).toHaveBeenLastCalledWith(252);
      await keys('1');
      expect(input.value).toBe('2521');

      await keys('{Control>}a{/Control}');
      await act(async () => {
        await userEvent.paste();
      });
      expect(input.value).toBe('2');
      expect(onCommit).toHaveBeenLastCalledWith(2);
    });

    it('normalizes a valid draft on blur and keeps an incomplete one visible and invalid', async () => {
      const { input, onCommit } = await renderStateful(FLOAT, 0.5);

      await keys('0');
      expect(input.value).toBe('0.50');
      expect(onCommit).toHaveBeenLastCalledWith(0.5);
      await blur(input);
      expect(input.value).toBe('0.5');

      await act(async () => {
        await userEvent.click(input);
      });
      await keys('{Control>}a{/Control}-');
      expect(input.value).toBe('');
      expect(onCommit).toHaveBeenLastCalledWith(undefined);
      await blur(input);
      expect((input as HTMLInputElement).validity.badInput).toBe(true);
      expect(isInvalid(input)).toBe(true);
      expect(onCommit).toHaveBeenLastCalledWith(undefined);
    });
  });

  describe('integer and bounds', () => {
    it('keeps a fractional or negative entry visible and invalid instead of correcting it', async () => {
      const fractional = await renderStateful(INTEGER, undefined);

      await keys('2.5');
      expect(fractional.input.value).toBe('2.5');
      expect(fractional.onCommit).toHaveBeenLastCalledWith(2.5);
      expect(isInvalid(fractional.input)).toBe(true);
      await keys('{Backspace}{Backspace}');
      expect(fractional.input.value).toBe('2');
      expect(fractional.onCommit).toHaveBeenLastCalledWith(2);
      expect(isInvalid(fractional.input)).toBe(false);

      const nonNegative = await renderStateful(fullTemplate('IntegerField', { minimum: 0 }), undefined);

      await keys('-7');
      expect(nonNegative.input.value).toBe('-7');
      expect(nonNegative.onCommit).toHaveBeenLastCalledWith(-7);
      expect(isInvalid(nonNegative.input)).toBe(true);
      await blur(nonNegative.input);
      expect(nonNegative.input.value).toBe('-7');
      expect(isInvalid(nonNegative.input)).toBe(true);
      await act(async () => {
        await userEvent.click(nonNegative.input);
      });
      await keys('{Home}{Delete}');
      expect(nonNegative.input.value).toBe('7');
      expect(nonNegative.onCommit).toHaveBeenLastCalledWith(7);
      expect(isInvalid(nonNegative.input)).toBe(false);

      const negativeFloat = await renderStateful(fullTemplate('FloatField', { minimum: 0 }), 4);

      await keys('{Home}-');
      expect(negativeFloat.input.value).toBe('-4');
      expect(negativeFloat.onCommit).toHaveBeenLastCalledWith(-4);
      expect(isInvalid(negativeFloat.input)).toBe(true);
    });

    it('edits a multi-digit integer at every position and completes a signed draft', async () => {
      const { input, onCommit } = await renderStateful(INTEGER, 1234);

      await keys('{Home}9');
      expect(input.value).toBe('91234');
      await keys('{End}{ArrowLeft}{ArrowLeft}{Backspace}');
      expect(input.value).toBe('9134');
      await keys('{Home}{ArrowRight}{Delete}');
      expect(input.value).toBe('934');
      expect(onCommit).toHaveBeenLastCalledWith(934);

      await keys('{Control>}a{/Control}56');
      expect(input.value).toBe('56');
      expect(onCommit).toHaveBeenLastCalledWith(56);

      await keys('{Control>}a{/Control}{Backspace}-');
      expect(input.value).toBe('');
      expect(onCommit).toHaveBeenLastCalledWith(undefined);
      await keys('12');
      expect(input.value).toBe('-12');
      expect(onCommit).toHaveBeenLastCalledWith(-12);
      expect(isInvalid(input)).toBe(false);

      await keys('{Control>}a{/Control}4{Backspace}');
      expect(input.value).toBe('');
      expect(onCommit).toHaveBeenLastCalledWith(undefined);
      await keys('4{Home}{Delete}');
      expect(input.value).toBe('');
      expect(document.activeElement).toBe(input);
    });

    it('marks out-of-range and off-step entries invalid but editable, then valid once corrected', async () => {
      const { input, onCommit } = await renderStateful(
        fullTemplate('FloatField', { exclusiveMaximum: 1, minimum: 0, multipleOf: 0.25 }),
        0.5
      );

      await keys('{Control>}a{/Control}1');
      expect(input.value).toBe('1');
      expect(isInvalid(input)).toBe(true);
      await keys('{Backspace}0.75');
      expect(input.value).toBe('0.75');
      expect(isInvalid(input)).toBe(false);

      await keys('{Backspace}');
      expect(input.value).toBe('0.7');
      expect(onCommit).toHaveBeenLastCalledWith(0.7);
      expect(isInvalid(input)).toBe(true);
      await keys('5');
      expect(input.value).toBe('0.75');
      expect(isInvalid(input)).toBe(false);

      await keys('{Home}-');
      expect(input.value).toBe('-0.75');
      expect(isInvalid(input)).toBe(true);
      await keys('{Home}{Delete}');
      expect(input.value).toBe('0.75');
      expect(isInvalid(input)).toBe(false);
    });
  });
});

describe('WorkflowFieldInput scalar collections', () => {
  let renderCount = 0;
  const list = (typeName: 'FloatField' | 'IntegerField' | 'StringField', overrides: Partial<FieldInputTemplate> = {}) =>
    fullTemplate(typeName, { type: { batch: false, cardinality: 'COLLECTION', name: typeName }, ...overrides });
  const renderList = async (template: FieldInputTemplate, initial: unknown, echo?: 'deferred' | 'sync') => {
    const onCommit = vi.fn();

    renderCount += 1;
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <StatefulField key={renderCount} echo={echo} initial={initial} template={template} onCommit={onCommit} />
        </ChakraProvider>
      );
    });

    return { onCommit };
  };
  // Browser tests render raw i18n keys, so interpolated row names collapse to one key: rows are found by position.
  const row = (index: number) =>
    host.querySelectorAll<HTMLInputElement>('input[aria-label="nodes.collectionItemLabel"]')[index - 1] ?? null;
  const rows = () => host.querySelectorAll('input[aria-label="nodes.collectionItemLabel"]').length;
  const button = (label: 'nodes.addItem' | 'common.clear') => findButton(label);
  const removeButton = (index: number) =>
    host.querySelectorAll<HTMLButtonElement>('button[aria-label="nodes.removeItem"]')[index - 1]!;
  const click = (element: HTMLElement) =>
    act(async () => {
      await userEvent.click(element);
    });
  const keys = (sequence: string) =>
    act(async () => {
      await userEvent.keyboard(sequence);
    });
  const isInvalid = (input: HTMLElement | null) => input?.getAttribute('aria-invalid') === 'true';

  it('appends, edits, and removes rows while each keystroke echoes back late', async () => {
    const { onCommit } = await renderList(list('IntegerField'), [1, 2], 'deferred');

    expect(rows()).toBe(2);
    await click(button('nodes.addItem'));
    expect(onCommit).toHaveBeenLastCalledWith([1, 2, 0]);
    expect(rows()).toBe(3);

    const third = row(3)!;

    await click(third);
    await keys('{Control>}a{/Control}42');
    expect(third.value).toBe('42');
    expect(onCommit).toHaveBeenLastCalledWith([1, 2, 42]);
    await keys('{Home}7');
    expect(third.value).toBe('742');
    expect(third.selectionStart).toBeNull(); // number inputs expose no caret; focus is the tell
    expect(document.activeElement).toBe(third);
    expect(onCommit).toHaveBeenLastCalledWith([1, 2, 742]);

    await click(removeButton(2));
    expect(onCommit).toHaveBeenLastCalledWith([1, 742]);
    expect(rows()).toBe(2);
    expect(row(2)?.value).toBe('742');
  });

  it('marks only the cleared or out-of-range row invalid and keeps the entry as typed', async () => {
    const { onCommit } = await renderList(list('IntegerField', { minimum: 0 }), [1, 2]);
    const first = row(1)!;

    await click(first);
    await keys('{Control>}a{/Control}{Backspace}-');
    expect(first.value).toBe('');
    expect(onCommit).toHaveBeenLastCalledWith([null, 2]);
    expect(isInvalid(first)).toBe(true);
    expect(isInvalid(row(2))).toBe(false);

    await keys('3');
    expect(first.value).toBe('-3');
    expect(onCommit).toHaveBeenLastCalledWith([-3, 2]);
    expect(isInvalid(first)).toBe(true);

    await keys('{Home}{Delete}');
    expect(first.value).toBe('3');
    expect(onCommit).toHaveBeenLastCalledWith([3, 2]);
    expect(isInvalid(first)).toBe(false);
  });

  it('clears a required list to empty and an optional one back to its absent default', async () => {
    const required = await renderList(list('StringField'), ['a', 'b']);

    await click(button('common.clear'));
    expect(required.onCommit).toHaveBeenLastCalledWith([]);
    expect(rows()).toBe(0);
    expect(Array.from(host.querySelectorAll('button')).map((element) => element.textContent)).not.toContain(
      'common.clear'
    );

    await click(button('nodes.addItem'));
    expect(required.onCommit).toHaveBeenLastCalledWith(['']);
    await click(row(1)!);
    await keys('hello');
    expect(required.onCommit).toHaveBeenLastCalledWith(['hello']);

    const optional = await renderList(list('FloatField', { required: false }), [0.5]);

    await click(removeButton(1));
    expect(optional.onCommit).toHaveBeenLastCalledWith(undefined);
    expect(rows()).toBe(0);
  });

  it('keeps keyboard focus in the list when a removal or clear unmounts the focused button', async () => {
    const { onCommit } = await renderList(list('IntegerField'), [1, 2, 3]);

    // Removing the last row unmounts its own button; focus steps back to the previous row's.
    await act(() => removeButton(3).focus());
    await keys('{Enter}');
    expect(onCommit).toHaveBeenLastCalledWith([1, 2]);
    expect(document.activeElement).toBe(removeButton(2));

    // Removing an earlier row keeps the button at that position, now owning the next item.
    await act(() => removeButton(1).focus());
    await keys('{Enter}');
    expect(onCommit).toHaveBeenLastCalledWith([2]);
    expect(document.activeElement).toBe(removeButton(1));

    await keys('{Enter}');
    expect(onCommit).toHaveBeenLastCalledWith([]);
    expect(document.activeElement).toBe(button('nodes.addItem'));

    await keys('{Enter}{Enter}');
    expect(rows()).toBe(2);
    await act(() => button('common.clear').focus());
    await keys('{Enter}');
    expect(rows()).toBe(0);
    expect(document.activeElement).toBe(button('nodes.addItem'));
  });
});

describe('WorkflowFieldInput record pickers', () => {
  const STYLE_PRESET = fullTemplate('StringField', {
    title: 'Style Preset',
    type: { batch: false, cardinality: 'SINGLE', name: 'StylePresetField' },
  });
  const SYSTEM_PROMPT = fullTemplate('StringField', {
    title: 'System Prompt',
    type: { batch: false, cardinality: 'SINGLE', name: 'SystemPromptField' },
  });
  const presetDto = (id: string, name: string) => ({
    id,
    image: null,
    is_public: false,
    name,
    preset_data: { negative_prompt: '', positive_prompt: '{prompt}, cinematic' },
    type: 'user',
    user_id: 'user-1',
  });
  const promptDto = (id: string, name: string) => ({
    content: 'Be helpful.',
    id,
    is_public: true,
    max_tokens: null,
    name,
    user_id: 'user-1',
  });
  const combobox = () => host.querySelector<HTMLInputElement>('input[role="combobox"]');
  const option = (label: string) =>
    Array.from(document.querySelectorAll<HTMLElement>('[data-scope="combobox"][data-part="item"]')).find((item) =>
      item.textContent?.includes(label)
    );

  it('picks a style preset by name and clears it back to an absent value', async () => {
    workflowApiMock.apiFetchJson.mockResolvedValue([
      presetDto('preset-1', 'Cinematic'),
      presetDto('preset-2', 'Anime'),
    ]);
    const onChange = vi.fn();

    await renderField(STYLE_PRESET, undefined, onChange);
    await vi.waitFor(() => expect(combobox()?.placeholder).toBe('nodes.stylePresetSearch'));
    expect(host.querySelector('button[aria-label="nodes.stylePresetClear"]')).toBeNull();

    await act(() => userEvent.click(combobox()!));
    await vi.waitFor(() => expect(option('Anime')).toBeDefined());
    await act(() => userEvent.click(option('Anime')!));
    expect(onChange).toHaveBeenLastCalledWith({ style_preset_id: 'preset-2' });

    await renderField(STYLE_PRESET, { style_preset_id: 'preset-2' }, onChange);
    await vi.waitFor(() => expect(combobox()?.value).toBe('Anime'));
    expect(combobox()?.getAttribute('aria-invalid')).toBeNull();
    await act(() =>
      userEvent.click(host.querySelector<HTMLButtonElement>('button[aria-label="nodes.stylePresetClear"]')!)
    );
    expect(onChange).toHaveBeenLastCalledWith(undefined);
    // The clear button leaves with the value; focus lands on the input rather than falling to the page.
    expect(document.activeElement).toBe(combobox());
  });

  it('keeps a system prompt id the list no longer holds and flags it as missing', async () => {
    workflowApiMock.apiFetchJson.mockResolvedValue([promptDto('prompt-1', 'Helpful')]);
    const onChange = vi.fn();

    await renderField(SYSTEM_PROMPT, { system_prompt_id: 'prompt-gone' }, onChange);
    await vi.waitFor(() => expect(combobox()?.value).toBe('nodes.systemPromptMissing'));
    expect(combobox()?.getAttribute('aria-invalid')).toBe('true');
    // The reason is linked to the input so assistive tech reads it with the invalid state.
    const describedBy = combobox()?.getAttribute('aria-describedby');
    expect(describedBy).toBeTruthy();
    expect(document.getElementById(describedBy!)?.textContent).toBe('nodes.systemPromptMissing');
    expect(onChange).not.toHaveBeenCalled();

    // The live record is still offered next to the missing one.
    await act(() => userEvent.click(combobox()!));
    await vi.waitFor(() => expect(option('Helpful')).toBeDefined());
    await act(() => userEvent.click(option('Helpful')!));
    expect(onChange).toHaveBeenLastCalledWith({ system_prompt_id: 'prompt-1' });
  });

  it('offers a retry when the list fails to load', async () => {
    workflowApiMock.apiFetchJson.mockRejectedValue(new Error('offline'));

    await renderField(SYSTEM_PROMPT, undefined, vi.fn());
    await vi.waitFor(() =>
      expect(host.querySelector<HTMLButtonElement>('button[aria-label="common.retry"]')).not.toBeNull()
    );
    expect(host.textContent).toContain('nodes.recordListFailed');

    workflowApiMock.apiFetchJson.mockResolvedValue([promptDto('prompt-1', 'Helpful')]);
    await act(() => userEvent.click(host.querySelector<HTMLButtonElement>('button[aria-label="common.retry"]')!));
    await vi.waitFor(() => expect(host.querySelector('button[aria-label="common.retry"]')).toBeNull());
    await act(() => userEvent.click(combobox()!));
    await vi.waitFor(() => expect(option('Helpful')).toBeDefined());
  });
});

describe('WorkflowFieldInput generators', () => {
  const FLOAT_GENERATOR = fullTemplate('FloatField', {
    default: { count: 10, start: 0, step: 0.1, type: 'float_generator_arithmetic_sequence' },
    input: 'direct',
    title: 'Generator Type',
    type: { batch: false, cardinality: 'SINGLE', name: 'FloatGeneratorField' },
  });
  const STRING_GENERATOR = fullTemplate('StringField', {
    default: { input: 'foo,bar,baz,qux', splitOn: ',', type: 'string_generator_parse_string' },
    input: 'direct',
    title: 'Generator Type',
    type: { batch: false, cardinality: 'SINGLE', name: 'StringGeneratorField' },
  });
  const IMAGE_GENERATOR = fullTemplate('StringField', {
    default: { category: 'images', type: 'image_generator_images_from_board' },
    input: 'direct',
    title: 'Generator Type',
    type: { batch: false, cardinality: 'SINGLE', name: 'ImageGeneratorField' },
  });
  let renderCount = 0;
  const renderGenerator = async (template: FieldInputTemplate, initial: unknown) => {
    const onCommit = vi.fn();

    renderCount += 1;
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <QueryClientProvider client={queryClient}>
            <StatefulField key={renderCount} initial={initial} template={template} onCommit={onCommit} />
          </QueryClientProvider>
        </ChakraProvider>
      );
    });
    await vi.waitFor(() => expect(host.querySelector('[data-scope="select"][data-part="trigger"]')).not.toBeNull());

    return { onCommit };
  };
  const numberInput = (suffix: string) =>
    host.querySelector<HTMLInputElement>(`input[type="number"][id$="${suffix}-number-input"]`)!;
  const selectVariant = async (label: string) => {
    await act(() => userEvent.click(host.querySelector('[data-scope="select"][data-part="trigger"]')!));
    const item = Array.from(document.querySelectorAll<HTMLElement>('[data-scope="select"][data-part="item"]')).find(
      (candidate) => candidate.textContent?.includes(label)
    )!;
    await act(() => userEvent.click(item));
  };

  it('previews an arithmetic sequence, commits edited settings as typed, and resets on a variant switch', async () => {
    const { onCommit } = await renderGenerator(FLOAT_GENERATOR, {
      count: 3,
      start: 1,
      step: 0.5,
      type: 'float_generator_arithmetic_sequence',
    });

    expect(host.textContent).toContain('1, 1.5, 2');

    await act(() => numberInput('-count').focus());
    await act(() => userEvent.keyboard('{Control>}a{/Control}4'));
    expect(onCommit).toHaveBeenLastCalledWith({
      count: 4,
      start: 1,
      step: 0.5,
      type: 'float_generator_arithmetic_sequence',
    });
    // The preview holds the pre-edit value through the typing burst, then settles.
    await vi.waitFor(() => expect(host.textContent).toContain('1, 1.5, 2, 2.5'), { timeout: 2000 });

    await selectVariant('nodes.parseString');
    expect(onCommit).toHaveBeenLastCalledWith({
      input: '0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1',
      splitOn: ',',
      type: 'float_generator_parse_string',
    });
  });

  it('shows a random draw as a count until a seed pins it, and refuses a count the queue cannot take', async () => {
    const { onCommit } = await renderGenerator(FLOAT_GENERATOR, {
      count: 4,
      max: 9,
      min: 2,
      seed: null,
      type: 'float_generator_random_distribution_uniform',
    });

    expect(host.textContent).toContain('<nodes.generatorNRandomValues>');
    expect(numberInput('-seed').disabled).toBe(true);
    expect(numberInput('-seed').getAttribute('aria-label')).toBe('nodes.generatorSeed');

    // A count below its minimum commits as typed, is flagged, and names its rule; the other settings survive.
    await act(() => numberInput('-count').focus());
    await act(() => userEvent.keyboard('{Control>}a{/Control}0'));
    expect(numberInput('-count').value).toBe('0');
    expect(onCommit).toHaveBeenLastCalledWith({
      count: 0,
      max: 9,
      min: 2,
      seed: null,
      type: 'float_generator_random_distribution_uniform',
    });
    expect(numberInput('-count').getAttribute('aria-invalid')).toBe('true');
    expect(numberInput('-min').getAttribute('aria-invalid')).toBeNull();
    expect(host.textContent).toContain('Count must be at least 1.');
    expect(numberInput('-min').value).toBe('2');
    await act(() => userEvent.keyboard('5'));
    expect(onCommit).toHaveBeenLastCalledWith({
      count: 5,
      max: 9,
      min: 2,
      seed: null,
      type: 'float_generator_random_distribution_uniform',
    });

    await act(() => userEvent.click(host.querySelector('[data-scope="checkbox"][data-part="control"]')!));
    expect(onCommit).toHaveBeenLastCalledWith({
      count: 5,
      max: 9,
      min: 2,
      seed: 0,
      type: 'float_generator_random_distribution_uniform',
    });
    expect(numberInput('-seed').disabled).toBe(false);
    expect(host.textContent).not.toContain('<nodes.generatorNRandomValues>');
    expect(host.textContent).toMatch(/\d+(\.\d+)?, \d+(\.\d+)?/);

    await act(() => numberInput('-count').focus());
    await act(() => userEvent.keyboard('{Control>}a{/Control}20000'));
    await vi.waitFor(() => expect(host.textContent).toContain('nodes.generatorTooMany'), { timeout: 2000 });

    // Clearing leaves the setting empty and flagged, on blur too, instead of restoring the last count.
    await act(() => userEvent.keyboard('{Control>}a{/Control}{Backspace}'));
    expect(onCommit).toHaveBeenLastCalledWith({
      count: null,
      max: 9,
      min: 2,
      seed: 0,
      type: 'float_generator_random_distribution_uniform',
    });
    expect(host.textContent).toContain('Count is empty.');
    await act(() => numberInput('-count').blur());
    expect(numberInput('-count').value).toBe('');
    expect(numberInput('-count').getAttribute('aria-invalid')).toBe('true');

    // A fractional pinned seed is kept and flagged rather than dropped back to a random draw.
    await act(() => numberInput('-seed').focus());
    await act(() => userEvent.keyboard('{Control>}a{/Control}2.5'));
    expect(onCommit).toHaveBeenLastCalledWith(expect.objectContaining({ seed: 2.5 }));
    expect(numberInput('-seed').getAttribute('aria-invalid')).toBe('true');
    expect(numberInput('-seed').disabled).toBe(false);
  });

  it('loads a parse-string source from a file and previews the split values', async () => {
    const { onCommit } = await renderGenerator(STRING_GENERATOR, {
      input: 'a|b',
      splitOn: '|',
      type: 'string_generator_parse_string',
    });

    expect(host.textContent).toContain('a, b');

    const fileInput = host.querySelector<HTMLInputElement>('input[type="file"]')!;

    await act(() => userEvent.upload(fileInput, new File(['x|y|z'], 'values.txt', { type: 'text/plain' })));
    await vi.waitFor(() =>
      expect(onCommit).toHaveBeenLastCalledWith({ input: 'x|y|z', splitOn: '|', type: 'string_generator_parse_string' })
    );
    await vi.waitFor(() => expect(host.textContent).toContain('x, y, z'), { timeout: 2000 });

    const toast = vi.spyOn(toaster, 'create');

    await act(() =>
      userEvent.upload(fileInput, new File([new Uint8Array(129 * 1024)], 'big.txt', { type: 'text/plain' }))
    );
    await vi.waitFor(() => expect(toast).toHaveBeenCalledWith({ title: 'nodes.generatorFileTooLarge', type: 'error' }));
    expect(onCommit).toHaveBeenLastCalledWith({ input: 'x|y|z', splitOn: '|', type: 'string_generator_parse_string' });
    toast.mockRestore();
  });

  it('previews dynamic prompts from the backend and marks a board the gallery no longer has', async () => {
    workflowApiMock.apiFetchJson.mockImplementation((path: string) =>
      path.includes('dynamicprompts')
        ? Promise.resolve({ error: null, prompts: ['a cute dog', 'a cute cat'] })
        : Promise.resolve([
            {
              board_id: 'b1',
              board_name: 'Portraits',
              image_count: 2,
              asset_count: 0,
              video_count: 0,
              asset_video_count: 0,
              archived: false,
            },
          ])
    );
    await renderGenerator(STRING_GENERATOR, {
      input: 'a cute {dog|cat}',
      maxPrompts: 5,
      type: 'string_generator_dynamic_prompts_combinatorial',
    });
    await vi.waitFor(() => expect(host.textContent).toContain('a cute dog, a cute cat'));

    await renderGenerator(IMAGE_GENERATOR, {
      board_id: 'gone',
      category: 'assets',
      type: 'image_generator_images_from_board',
    });
    await vi.waitFor(() =>
      expect(host.querySelector<HTMLInputElement>('input[role="combobox"]')?.value).toBe('nodes.generatorBoardMissing')
    );
    expect(host.querySelector('input[role="combobox"]')?.getAttribute('aria-invalid')).toBe('true');
  });
});
