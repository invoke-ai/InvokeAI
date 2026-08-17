import { useCallback, useRef, type ChangeEvent } from 'react';

export const ACCEPTED_UPLOAD_EXTENSIONS = 'image/png,image/jpeg,image/webp,video/mp4,.png,.jpg,.jpeg,.webp,.mp4';
export const UPLOAD_INPUT_STYLE = { display: 'none' } as const;

/** The hidden-input picker every gallery upload trigger shares: change extracts files, resets the input (so re-picking the same file fires), and forwards. */
export const useGalleryUploadInput = (onUploadFiles: (files: File[]) => Promise<void> | void) => {
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
      accept: ACCEPTED_UPLOAD_EXTENSIONS,
      multiple: true as const,
      onChange,
      ref: inputRef,
      style: UPLOAD_INPUT_STYLE,
      type: 'file' as const,
    },
    openPicker,
  };
};
