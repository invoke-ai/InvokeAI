import { FormControl, FormLabel, Select } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { formElementNodeFieldDataChanged } from 'features/nodes/store/nodesSlice';
import { getDefaultStringOption, type NodeFieldStringSettings, zStringComponent } from 'features/nodes/types/workflow';
import { omit } from 'lodash-es';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
  settings: NodeFieldStringSettings;
};

export const NodeFieldElementStringSettings = memo(({ id, settings }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const onChangeComponent = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      const component = zStringComponent.parse(e.target.value);
      if (component === settings.component) {
        // no change
        return;
      }
      if (component === 'dropdown') {
        // if the component is changing to dropdown, we need to set the options
        const newSettings: NodeFieldStringSettings = {
          ...settings,
          component,
          options: [getDefaultStringOption()],
        };
        dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: newSettings } }));
        return;
      } else {
        // if the component is changing from dropdown, we need to remove the options
        const newSettings: NodeFieldStringSettings = omit({ ...settings, component }, 'options');
        dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: newSettings } }));
      }
    },
    [settings, dispatch, id]
  );

  return (
    <FormControl>
      <FormLabel flex={1}>{t('workflows.builder.component')}</FormLabel>
      <Select value={settings.component} onChange={onChangeComponent} size="sm">
        <option value="input">{t('workflows.builder.singleLine')}</option>
        <option value="textarea">{t('workflows.builder.multiLine')}</option>
        <option value="dropdown">{t('workflows.builder.dropdown')}</option>
      </Select>
    </FormControl>
  );
});
NodeFieldElementStringSettings.displayName = 'NodeFieldElementStringSettings';
