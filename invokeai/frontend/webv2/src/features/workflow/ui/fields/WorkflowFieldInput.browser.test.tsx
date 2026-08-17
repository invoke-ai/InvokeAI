import type { FieldInputTemplate } from '@features/workflow/contracts';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { WorkflowFieldInput } from './WorkflowFieldInput';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const uploadImageMock = vi.fn();
const uploadVideoMock = vi.fn();
const resolveItemMock = vi.fn();

vi.mock('@features/gallery', () => ({
  formatGalleryVideoDuration: (seconds: number) => `${seconds}s`,
  galleryDestinations: { list: () => Promise.resolve([]) },
  galleryItems: { resolve: (...args: unknown[]) => resolveItemMock(...args) },
  galleryTransfers: {
    upload: (...args: unknown[]) => uploadImageMock(...args),
    uploadVideo: (...args: unknown[]) => uploadVideoMock(...args),
  },
}));

const galleryValues: Record<string, unknown> = {};
const graphNodes: unknown[] = [];
const graphEdges: unknown[] = [];
const projectSnapshot = { galleryValues, id: 'project-1', projectGraph: { edges: graphEdges, nodes: graphNodes } };

vi.mock('@features/workflow/ui/WorkflowUiContext', () => ({
  useWorkflowProjectSelector: (selector: (project: typeof projectSnapshot) => unknown) => selector(projectSnapshot),
  useWorkflowUi: () => ({ project: { getSnapshot: () => projectSnapshot } }),
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

const FRAME_INDEX_TEMPLATE = {
  name: 'frame_index',
  title: 'Frame Index',
  type: { cardinality: 'SINGLE', name: 'IntegerField' },
  uiComponent: 'video-frame-index',
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
  queryClient.clear();
  delete galleryValues.selectedImage;
  graphNodes.length = 0;
  graphEdges.length = 0;
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

const queryClient = new QueryClient();

const renderField = async (
  template: FieldInputTemplate,
  value: unknown,
  onChange: (value: unknown) => void,
  nodeId?: string
) => {
  await act(() => {
    root.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient}>
          <DndContext>
            <WorkflowFieldInput nodeId={nodeId} template={template} value={value} onChange={onChange} />
          </DndContext>
        </QueryClientProvider>
      </ChakraProvider>
    );
  });
};

const findButton = (label: string): HTMLButtonElement => {
  const button = Array.from(host.querySelectorAll('button')).find((el) => el.textContent?.includes(label));

  if (!button) {
    throw new Error(`Button "${label}" not found`);
  }

  return button;
};

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

describe('WorkflowFieldInput media inputs', () => {
  it('renders a direct-input widget for VideoField instead of falling back to connection-only', async () => {
    await renderField(VIDEO_TEMPLATE, undefined, vi.fn());

    expect(host.textContent).not.toContain('Connection only');
    expect(host.textContent).toContain('Drop a video here');
    expect(findButton('Use gallery selection').disabled).toBe(true);
    expect(findButton('Upload').disabled).toBe(false);
    expect(host.querySelector<HTMLInputElement>('input[type="file"]')?.accept).toBe('video/*');
  });

  it('adopts a selected gallery video and clears it', async () => {
    galleryValues.selectedImage = SELECTED_GALLERY_VIDEO;
    const onChange = vi.fn();

    await renderField(VIDEO_TEMPLATE, undefined, onChange);

    const useSelection = findButton('Use gallery selection');

    expect(useSelection.disabled).toBe(false);
    await act(() => useSelection.click());
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

  it('does not offer a selected gallery image to a video field', async () => {
    galleryValues.selectedImage = { ...SELECTED_GALLERY_VIDEO, kind: 'image' };

    await renderField(VIDEO_TEMPLATE, undefined, vi.fn());

    expect(findButton('Use gallery selection').disabled).toBe(true);
  });

  it('keeps COLLECTION media fields connection-only (the widget would write a bare object into a list)', async () => {
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
