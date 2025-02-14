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
import { useInputFieldDescription } from 'features/nodes/hooks/useInputFieldDescription';
import { fieldDescriptionChanged } from 'features/nodes/store/nodesSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiNoteBold } from 'react-icons/pi';

type Props = {
  nodeId: string;
  fieldName: string;
};

export const InputFieldDescriptionPopover = memo(({ nodeId, fieldName }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const description = useInputFieldDescription(nodeId, fieldName);
  const onChange = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      dispatch(fieldDescriptionChanged({ nodeId, fieldName, val: e.target.value }));
    },
    [dispatch, fieldName, nodeId]
  );

  return (
    <Popover>
      <PopoverTrigger>
        <IconButton
          variant="ghost"
          tooltip={t('nodes.description')}
          aria-label={t('nodes.description')}
          icon={<PiNoteBold />}
          pointerEvents="auto"
          size="xs"
        />
      </PopoverTrigger>
      <PopoverContent p={2} w={256}>
        <FormControl orientation="vertical">
          <FormLabel>{t('nodes.description')}</FormLabel>
          <Textarea
            className="nodrag nopan nowheel"
            fontSize="sm"
            value={description ?? ''}
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

InputFieldDescriptionPopover.displayName = 'InputFieldDescriptionPopover';
