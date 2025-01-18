import { IconButton, type IconButtonProps } from '@invoke-ai/ui-library';
import { toast } from 'features/toast/toast';
import { isString } from 'lodash-es';
import { memo, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { PiUploadFill } from 'react-icons/pi';

type Props = Omit<IconButtonProps, 'aria-label'> & {
  onLoadFile: (value: string) => void;
  maxSize?: number;
};

const DEFAULT_MAX_SIZE = 1024 * 128; // 128KB, we don't want to load huge files into node values...

export const LoadTextFromFileIconButton = memo(({ onLoadFile, maxSize = DEFAULT_MAX_SIZE, ...rest }: Props) => {
  const onDropAccepted = useCallback(
    (files: File[]) => {
      const file = files[0];
      if (!file) {
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result;
        if (!isString(result)) {
          return;
        }
        onLoadFile(result);
      };
      reader.onerror = () => {
        toast({
          title: 'Failed to load file',
          status: 'error',
        });
      };
      reader.readAsText(file);
    },
    [onLoadFile]
  );

  const { getInputProps, getRootProps } = useDropzone({
    accept: { 'text/csv': ['.csv'], 'text/plain': ['.txt'] },
    maxSize,
    onDropAccepted,
    noDrag: true,
    multiple: false,
  });
  return (
    <>
      <IconButton
        tooltip="Load from file"
        aria-label="Load from file"
        icon={<PiUploadFill />}
        variant="link"
        boxSize={8}
        minW={8}
        {...rest}
        {...getRootProps()}
      />
      <input {...getInputProps()} />
    </>
  );
});
LoadTextFromFileIconButton.displayName = 'LoadTextFromFileIconButton';
