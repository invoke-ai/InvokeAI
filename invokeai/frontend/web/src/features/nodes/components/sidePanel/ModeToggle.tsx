import { useCallback, useMemo } from 'react';
import { useAppDispatch, useAppSelector } from '../../../../app/store/storeHooks';
import { workflowModeChanged } from '../../store/workflowSlice';
import { Combobox, ComboboxOnChange } from '@invoke-ai/ui-library';
import { WorkflowMode } from '../../store/types';

const MODE_OPTIONS = [
  { value: 'view', label: 'View' },
  { value: 'edit', label: 'Edit' },
];

export const ModeToggle = () => {
  const dispatch = useAppDispatch();
  const mode = useAppSelector((s) => s.workflow.mode);

  const onChange = useCallback<ComboboxOnChange>((v) => {
    if (!v) {
      return;
    }
    dispatch(workflowModeChanged(v.value as WorkflowMode));
  }, []);

  const value = useMemo(() => MODE_OPTIONS.find((o) => o.value === mode), [mode]);

  return <Combobox value={value} onChange={onChange} options={MODE_OPTIONS} colorScheme="invokeBlue.300" />;
};
