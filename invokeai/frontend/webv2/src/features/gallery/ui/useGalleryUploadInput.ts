import { getGalleryUploadAccept } from '@features/gallery/core/items';
import { useCallback, useRef, type ChangeEvent } from 'react';

const ALL_UPLOAD_KINDS = ['image', 'video'] as const;
const UPLOAD_INPUT_STYLE = { display: 'none' } as const;

/**
 * The hidden-input picker every gallery upload trigger shares: change extracts files, resets the input (so re-picking
 * the same file fires) and forwards. The default `accept` is computed per call: in this lazy chunk, a module-scope call
 * into a core bundled elsewhere would run before that core initialised whenever the chunks import each other.
 */
export const useGalleryUploadInput = (
  onUploadFiles: (files: File[]) => Promise<unknown> | void,
  { accept = getGalleryUploadAccept(ALL_UPLOAD_KINDS), multiple = true }: { accept?: string; multiple?: boolean } = {}
) => {
  const inputRef = useRef<HTMLInputElement | null>(null);

  const onChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(event.currentTarget.files ?? []);

      event.currentTarget.value = '';

      if (files.length > 0) {
        void onUploadFiles(files);
      }
    },
    [onUploadFiles]
  );

  const openPicker = useCallback(() => inputRef.current?.click(), []);

  return {
    inputProps: {
      accept,
      multiple,
      onChange,
      ref: inputRef,
      style: UPLOAD_INPUT_STYLE,
      type: 'file' as const,
    },
    openPicker,
  };
};
