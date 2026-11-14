const carriesFiles = (event: DragEvent): boolean => event.dataTransfer?.types.includes('Files') ?? false;

// A dropped link or web image (`text/uri-list`) also opens in the tab unless the page takes it.
const carriesLink = (event: DragEvent): boolean => event.dataTransfer?.types.includes('text/uri-list') ?? false;

const isFileInput = (target: EventTarget | null): boolean =>
  target instanceof HTMLInputElement && target.type === 'file';

// Text fields and editors insert a dropped link as text.
const isTextTarget = (target: EventTarget | null): boolean =>
  target instanceof HTMLInputElement ||
  target instanceof HTMLTextAreaElement ||
  (target instanceof HTMLElement && target.isContentEditable);

/**
 * Whether the browser would open the drag in the tab, as nothing on the page took it. A drag carrying files is a file
 * drag whatever text it also carries: only a native file input takes those, and a text field dropped on would let the
 * browser open the file. A link without files is left to text fields and editors, which insert it.
 */
const opensUnhandled = (event: DragEvent): boolean => {
  if (event.defaultPrevented) {
    return false;
  }
  if (carriesFiles(event)) {
    return !isFileInput(event.target);
  }
  return carriesLink(event) && !isTextTarget(event.target);
};

/**
 * Keeps a file or link dragged in from the desktop or another window from replacing the app. Surfaces that accept a
 * drop take it first and are left alone, as are native file inputs, and text fields for links; anywhere else the
 * drag shows that it cannot be dropped, and a drop that still arrives is swallowed. Plain text drags never navigate
 * and stay untouched.
 */
export const preventFileDropNavigation = (): (() => void) => {
  const handleDragOver = (event: DragEvent) => {
    if (opensUnhandled(event)) {
      event.preventDefault();
      event.dataTransfer!.dropEffect = 'none';
    }
  };
  const handleDrop = (event: DragEvent) => {
    if (opensUnhandled(event)) {
      event.preventDefault();
    }
  };
  window.addEventListener('dragover', handleDragOver);
  window.addEventListener('drop', handleDrop);
  return () => {
    window.removeEventListener('dragover', handleDragOver);
    window.removeEventListener('drop', handleDrop);
  };
};
