import {
  FormControl,
  FormLabel,
  IconButton,
  Popover,
  PopoverContent,
  PopoverTrigger,
  Textarea,
} from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useInputFieldNotes } from 'features/nodes/hooks/useInputFieldNotes';
import { fieldNotesChanged } from 'features/nodes/store/nodesSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiNoteBold } from 'react-icons/pi';

type Props = {
  nodeId: string;
  fieldName: string;
};

export const InputFieldNotesIconButtonEditable = memo(({ nodeId, fieldName }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const notes = useInputFieldNotes(nodeId, fieldName);
  const onChange = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      dispatch(fieldNotesChanged({ nodeId, fieldName, val: e.target.value }));
    },
    [dispatch, fieldName, nodeId]
  );

  return (
    <Popover>
      <PopoverTrigger>
        <IconButton
          variant="ghost"
          tooltip={t('nodes.notes')}
          aria-label={t('nodes.notes')}
          icon={<PiNoteBold />}
          pointerEvents="auto"
          size="xs"
        />
      </PopoverTrigger>
      <PopoverContent p={2} w={256}>
        <FormControl orientation="vertical">
          <FormLabel>{t('nodes.notes')}</FormLabel>
          <Textarea
            className="nodrag nopan nowheel"
            fontSize="sm"
            value={notes ?? ''}
            onChange={onChange}
            p={2}
            resize="none"
            rows={5}
          />
        </FormControl>
      </PopoverContent>
    </Popover>
  );
});

InputFieldNotesIconButtonEditable.displayName = 'InputFieldNotesIconButtonEditable';
