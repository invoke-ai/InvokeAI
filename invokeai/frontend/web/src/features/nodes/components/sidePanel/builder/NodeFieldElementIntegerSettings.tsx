import { FormControl, FormLabel, Select } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { formElementNodeFieldDataChanged } from 'features/nodes/store/workflowSlice';
import type { NodeFieldIntegerConfig } from 'features/nodes/types/workflow';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const NodeFieldElementIntegerConfig = memo(({ id, config }: { id: string; config: NodeFieldIntegerConfig }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const onChangeComponent = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      const newConfig: NodeFieldIntegerConfig = {
        ...config,
        component: e.target.value as NodeFieldIntegerConfig['component'],
      };
      dispatch(formElementNodeFieldDataChanged({ id, changes: { config: newConfig } }));
    },
    [config, dispatch, id]
  );

  return (
    <FormControl>
      <FormLabel flex={1}>{t('workflows.builder.component')}</FormLabel>
      <Select value={config.component} onChange={onChangeComponent} size='sm'>
        <option value="input">{t('workflows.builder.input')}</option>
        <option value="slider">{t('workflows.builder.slider')}</option>
        <option value="input-and-slider">{t('workflows.builder.both')}</option>
      </Select>
    </FormControl>
  );
});
NodeFieldElementIntegerConfig.displayName = 'NodeFieldElementIntegerConfig';
