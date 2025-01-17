import {
  Box,
  FormControl,
  FormLabel,
  IconButton,
  Popover,
  PopoverContent,
  PopoverTrigger,
  Text,
  Textarea,
} from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useFieldNotes } from 'features/nodes/hooks/useFieldNotes';
import { fieldNotesChanged } from 'features/nodes/store/nodesSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiNoteBold } from 'react-icons/pi';

type Props = {
  nodeId: string;
  fieldName: string;
  readOnly?: boolean;
};

export const FieldNotesIconButton = memo(({ nodeId, fieldName, readOnly }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const notes = useFieldNotes(nodeId, fieldName);
  const onChange = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      dispatch(fieldNotesChanged({ nodeId, fieldName, val: e.target.value }));
    },
    [dispatch, fieldName, nodeId]
  );

  if (readOnly && !notes?.trim()) {
    return null;
  }

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
          {readOnly && (
            <Box borderWidth={1} borderRadius="base" p={2} w="full" h="full">
              <Text fontSize="sm">{notes}</Text>
            </Box>
          )}
          {!readOnly && (
            <Textarea
              className="nodrag nopan nowheel"
              fontSize="sm"
              value={notes ?? ''}
              onChange={onChange}
              p={2}
              resize="none"
              rows={5}
            />
          )}
        </FormControl>
      </PopoverContent>
    </Popover>
  );
});

FieldNotesIconButton.displayName = 'FieldNotesIconButton';
