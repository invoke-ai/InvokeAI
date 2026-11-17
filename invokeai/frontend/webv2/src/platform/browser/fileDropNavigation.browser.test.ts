import { afterEach, describe, expect, it, vi } from 'vitest';
import { commands } from 'vitest/browser';

import { preventFileDropNavigation } from './fileDropNavigation';
import { TINY_PNG_BASE64 } from './nativeDrag.testing';

const PNG_FILE = {
  base64: TINY_PNG_BASE64,
  name: 'photo.png',
};
const LINK = 'https://example.com/photo.png';

/*
 * A drop driven through CDP never navigates the headless test frame, guard or not, so these tests read what the
 * guard did to each event: a drag it takes is cancelled (`dragover` prevented, with no drop effect, so the browser
 * dispatches no drop and opens nothing), and one it leaves reaches the browser's default.
 */
describe('preventFileDropNavigation', () => {
  const cleanups: (() => void)[] = [];

  afterEach(() => {
    cleanups.splice(0).forEach((cleanup) => cleanup());
  });

  const mount = <T extends HTMLElement>(element: T): T => {
    Object.assign(element.style, { height: '120px', left: '20px', position: 'fixed', top: '20px', width: '120px' });
    document.body.append(element);
    cleanups.push(() => element.remove());
    return element;
  };
  /**
   * Installs the guard, then a recorder after it noting whether the guard took each drag event (`taken`) or left it to
   * the browser (`open`). The recorder lets everything it sees through, so a text field still takes what it is given.
   */
  const installGuard = () => {
    const seen: string[] = [];
    cleanups.push(preventFileDropNavigation());
    const record = (event: DragEvent) => {
      seen.push(`${event.type}:${event.defaultPrevented ? 'taken' : 'open'}`);
    };
    window.addEventListener('dragover', record);
    window.addEventListener('drop', record);
    cleanups.push(() => {
      window.removeEventListener('dragover', record);
      window.removeEventListener('drop', record);
    });
    return seen;
  };
  const dragOnto = (element: HTMLElement, data: { files?: (typeof PNG_FILE)[]; text?: string; uriList?: string }) => {
    const rect = element.getBoundingClientRect();
    return commands.nativeDrag({
      ...data,
      frameWidth: window.innerWidth,
      points: [{ x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 }],
    });
  };
  const dragOvers = (seen: string[]) => [...new Set(seen.filter((entry) => entry.startsWith('dragover')))];
  /** A drop the test dispatches itself: whether the guard cancelled it, for a drop a browser would still deliver. */
  const dropTaken = (target: HTMLElement, types: Record<string, string>, file?: File) => {
    const dataTransfer = new DataTransfer();
    if (file) {
      dataTransfer.items.add(file);
    }
    for (const [type, value] of Object.entries(types)) {
      dataTransfer.setData(type, value);
    }
    const event = new DragEvent('drop', { bubbles: true, cancelable: true, dataTransfer });
    target.dispatchEvent(event);
    return event.defaultPrevented;
  };
  /** One text field at the drag's point: an input, a textarea (as a prompt field is) or a contenteditable editor. */
  const textField = (kind: 'editor' | 'input' | 'textarea') =>
    mount(
      kind === 'editor'
        ? Object.assign(document.createElement('div'), { contentEditable: 'true' })
        : kind === 'input'
          ? Object.assign(document.createElement('input'), { type: 'text' })
          : document.createElement('textarea')
    );

  it('takes a file or link dragged over the app where nothing takes it, so the browser never opens it in the tab', async () => {
    const seen = installGuard();
    const plain = mount(document.createElement('div'));

    await dragOnto(plain, { files: [PNG_FILE] });
    expect(dragOvers(seen)).toEqual(['dragover:taken']);

    seen.length = 0;
    // An image dragged from another browser window often carries only its address.
    await dragOnto(plain, { uriList: LINK });
    expect(dragOvers(seen)).toEqual(['dragover:taken']);
  });

  it.each(['input', 'textarea', 'editor'] as const)(
    'takes a file dropped on a text %s, which cannot hold it, even when the drag also carries text',
    async (field) => {
      const seen = installGuard();
      const target = textField(field);

      await dragOnto(target, { files: [PNG_FILE] });
      expect(dragOvers(seen)).toEqual(['dragover:taken']);

      seen.length = 0;
      await dragOnto(target, { files: [PNG_FILE], text: 'photo.png', uriList: LINK });
      expect(dragOvers(seen)).toEqual(['dragover:taken']);
      expect(seen).not.toContain('drop:open');

      const file = new File(['x'], 'photo.png', { type: 'image/png' });
      expect(dropTaken(target, {}, file)).toBe(true);
      expect(dropTaken(target, { 'text/plain': 'photo.png', 'text/uri-list': LINK }, file)).toBe(true);
    }
  );

  it.each(['input', 'textarea', 'editor'] as const)('leaves a link dropped on a text %s to it', async (field) => {
    const seen = installGuard();
    const target = textField(field);

    await dragOnto(target, { uriList: LINK });

    expect(dragOvers(seen)).toEqual(['dragover:open']);
    expect(dropTaken(target, { 'text/uri-list': LINK })).toBe(false);
  });

  it('leaves plain text to the browser, which inserts it into a text field', async () => {
    const seen = installGuard();
    const input = textField('input') as HTMLInputElement;

    await dragOnto(input, { text: 'a red fox' });

    expect(seen).not.toContain('dragover:taken');
    await vi.waitFor(() => expect(input.value).toBe('a red fox'));
  });

  it('leaves a file to a native file input', async () => {
    const seen = installGuard();
    const input = mount(Object.assign(document.createElement('input'), { type: 'file' }));

    await dragOnto(input, { files: [PNG_FILE] });

    expect(dragOvers(seen)).toEqual(['dragover:open']);
  });

  it('leaves a drag a surface takes to that surface: its drop arrives with the file', async () => {
    installGuard();
    const surface = mount(document.createElement('div'));
    const dropped: string[] = [];
    surface.addEventListener('dragover', (event) => {
      event.preventDefault();
      event.dataTransfer!.dropEffect = 'copy';
    });
    surface.addEventListener('drop', (event) => {
      event.preventDefault();
      dropped.push(...[...event.dataTransfer!.files].map((file) => file.name));
    });

    await dragOnto(surface, { files: [PNG_FILE] });

    expect(dropped).toEqual(['photo.png']);
  });
});
