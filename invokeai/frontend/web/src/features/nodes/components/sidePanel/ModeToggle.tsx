import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { WorkflowMode } from 'features/nodes/store/types';
import { workflowModeChanged } from 'features/nodes/store/workflowSlice';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const ModeToggle = () => {
  const dispatch = useAppDispatch();
  const mode = useAppSelector((s) => s.workflow.mode);
  const { t } = useTranslation();

  const modeOptions = useMemo(() => {
    return [
      { value: 'view', label: t('nodes.viewMode') },
      { value: 'edit', label: t('nodes.editMode') },
    ];
  }, [t]);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!v) {
        return;
      }
      dispatch(workflowModeChanged(v.value as WorkflowMode));
    },
    [dispatch]
  );

  const value = useMemo(() => modeOptions.find((o) => o.value === mode), [mode, modeOptions]);

  return <Combobox value={value} onChange={onChange} options={modeOptions} colorScheme="invokeBlue.300" />;
};
