import { FormControl, FormLabel, Select } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { formElementNodeFieldDataChanged } from 'features/nodes/store/workflowSlice';
import { type NodeFieldIntegerSettings, zNumberComponent } from 'features/nodes/types/workflow';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const NodeFieldElementIntegerConfig = memo(
  ({ id, config }: { id: string; config: NodeFieldIntegerSettings }) => {
    const { t } = useTranslation();
    const dispatch = useAppDispatch();

    const onChangeComponent = useCallback(
      (e: ChangeEvent<HTMLSelectElement>) => {
        const newConfig: NodeFieldIntegerSettings = {
          ...config,
          component: zNumberComponent.parse(e.target.value),
        };
        dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: newConfig } }));
      },
      [config, dispatch, id]
    );

    return (
      <FormControl>
        <FormLabel flex={1}>{t('workflows.builder.component')}</FormLabel>
        <Select value={config.component} onChange={onChangeComponent} size="sm">
          <option value="number-input">{t('workflows.builder.numberInput')}</option>
          <option value="slider">{t('workflows.builder.slider')}</option>
          <option value="number-input-and-slider">{t('workflows.builder.both')}</option>
        </Select>
      </FormControl>
    );
  }
);
NodeFieldElementIntegerConfig.displayName = 'NodeFieldElementIntegerConfig';
