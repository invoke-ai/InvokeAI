import { FormControl, FormLabel, Select } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { formElementNodeFieldDataChanged } from 'features/nodes/store/workflowSlice';
import { getDefaultStringOption, type NodeFieldStringSettings, zStringComponent } from 'features/nodes/types/workflow';
import { omit } from 'lodash-es';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const NodeFieldElementStringSettings = memo(
  ({ id, config }: { id: string; config: NodeFieldStringSettings }) => {
    const { t } = useTranslation();
    const dispatch = useAppDispatch();

    const onChangeComponent = useCallback(
      (e: ChangeEvent<HTMLSelectElement>) => {
        const component = zStringComponent.parse(e.target.value);
        if (component === config.component) {
          // no change
          return;
        }
        if (component === 'dropdown') {
          // if the component is changing to dropdown, we need to set the choices
          const newConfig: NodeFieldStringSettings = {
            ...config,
            component,
            options: [getDefaultStringOption()],
          };
          dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: newConfig } }));
          return;
        }
        // if the component is changing from dropdown, we need to remove the choices
        const newConfig: NodeFieldStringSettings = omit({ ...config, component }, 'choices');
        dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: newConfig } }));
      },
      [config, dispatch, id]
    );

    return (
      <FormControl>
        <FormLabel flex={1}>{t('workflows.builder.component')}</FormLabel>
        <Select value={config.component} onChange={onChangeComponent} size="sm">
          <option value="input">{t('workflows.builder.singleLine')}</option>
          <option value="textarea">{t('workflows.builder.multiLine')}</option>
          <option value="dropdown">{t('workflows.builder.dropdown')}</option>
        </Select>
      </FormControl>
    );
  }
);
NodeFieldElementStringSettings.displayName = 'NodeFieldElementStringSettings';
