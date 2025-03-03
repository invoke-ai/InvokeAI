import { FormControl, FormLabel, IconButton, Spacer, Textarea } from '@invoke-ai/ui-library';
import { NO_DRAG_CLASS, NO_PAN_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import { toast } from 'features/toast/toast';
import { isString } from 'lodash-es';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiUploadFill } from 'react-icons/pi';

const MAX_SIZE = 1024 * 128; // 128KB, we don't want to load huge files into node values...

type Props = {
  value: string;
  onChange: (value: string) => void;
};

export const GeneratorTextareaWithFileUpload = memo(({ value, onChange }: Props) => {
  const { t } = useTranslation();
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
        onChange(result);
      };
      reader.onerror = () => {
        toast({
          title: 'Failed to load file',
          status: 'error',
        });
      };
      reader.readAsText(file);
    },
    [onChange]
  );

  const { getInputProps, getRootProps } = useDropzone({
    accept: { 'text/csv': ['.csv'], 'text/plain': ['.txt'] },
    maxSize: MAX_SIZE,
    onDropAccepted,
    noDrag: true,
    multiple: false,
  });

  const onChangeInput = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      onChange(e.target.value);
    },
    [onChange]
  );

  return (
    <FormControl orientation="vertical" position="relative" alignItems="stretch">
      <FormLabel m={0} display="flex" alignItems="center">
        {t('common.input')}
        <Spacer />
        <IconButton
          tooltip={t('nodes.generatorLoadFromFile')}
          aria-label={t('nodes.generatorLoadFromFile')}
          icon={<PiUploadFill />}
          variant="link"
          {...getRootProps()}
        />
        <input {...getInputProps()} />
      </FormLabel>
      <Textarea
        className={`${NO_DRAG_CLASS} ${NO_PAN_CLASS} ${NO_WHEEL_CLASS}`}
        value={value}
        onChange={onChangeInput}
        p={2}
        resize="none"
        rows={5}
        fontSize="sm"
      />
    </FormControl>
  );
});
GeneratorTextareaWithFileUpload.displayName = 'GeneratorTextareaWithFileUpload';
