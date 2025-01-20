import {
  FormControl,
  FormLabel,
  IconButton,
  Popover,
  PopoverContent,
  PopoverTrigger,
  Select,
} from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useInputFieldIsExposed } from 'features/nodes/hooks/useInputFieldIsExposed';
import { useInputFieldLinearViewConfig } from 'features/nodes/hooks/useInputFieldLinearViewConfig';
import { fieldLinearViewConfigChanged } from 'features/nodes/store/nodesSlice';
import type { FieldInputInstance } from 'features/nodes/types/field';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiWrenchFill } from 'react-icons/pi';

type Props = {
  nodeId: string;
  fieldName: string;
};

const parseNotesDisplay = (
  notesDisplay: string
): NonNullable<FieldInputInstance['linearViewConfig']>['notesDisplay'] => {
  switch (notesDisplay) {
    case 'none':
      return 'none';
    case 'helper-text':
      return 'helper-text';
    case 'icon-with-popover':
      return 'icon-with-popover';
    default:
      return 'none';
  }
};

export const InputFieldLinearViewConfigIconButton = memo(({ nodeId, fieldName }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const isExposed = useInputFieldIsExposed(nodeId, fieldName);
  const linearViewConfig = useInputFieldLinearViewConfig(nodeId, fieldName);
  const onChangeNotesDisplay = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      const notesDisplay = parseNotesDisplay(e.target.value);
      dispatch(
        fieldLinearViewConfigChanged({ nodeId, fieldName, linearViewConfig: { ...linearViewConfig, notesDisplay } })
      );
    },
    [dispatch, fieldName, linearViewConfig, nodeId]
  );

  if (!isExposed) {
    return null;
  }

  return (
    <Popover>
      <PopoverTrigger>
        <IconButton
          variant="ghost"
          tooltip="Linear View Config"
          aria-label="Linear View Config"
          icon={<PiWrenchFill />}
          pointerEvents="auto"
          size="xs"
        />
      </PopoverTrigger>
      <PopoverContent p={2} w={256}>
        <FormControl orientation="vertical">
          <FormLabel>{t('nodes.notesDisplay')}</FormLabel>
          <Select value={linearViewConfig?.notesDisplay ?? 'none'} onChange={onChangeNotesDisplay}>
            <option value="none">{t('common.none')}</option>
            <option value="helper-text">{t('nodes.helperText')}</option>
            <option value="icon-with-popover">{t('nodes.iconWithPopover')}</option>
          </Select>
        </FormControl>
      </PopoverContent>
    </Popover>
  );
});

InputFieldLinearViewConfigIconButton.displayName = 'InputFieldLinearViewConfigIconButton';
