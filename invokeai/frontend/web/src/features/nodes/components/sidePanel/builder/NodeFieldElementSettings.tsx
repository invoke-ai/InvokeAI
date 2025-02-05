import {
  FormControl,
  FormLabel,
  IconButton,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Switch,
} from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { NodeFieldElementFloatSettings } from 'features/nodes/components/sidePanel/builder/NodeFieldElementFloatSettings';
import { NodeFieldElementIntegerConfig } from 'features/nodes/components/sidePanel/builder/NodeFieldElementIntegerSettings';
import { formElementNodeFieldDataChanged } from 'features/nodes/store/workflowSlice';
import type { NodeFieldElement } from 'features/nodes/types/workflow';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiWrenchFill } from 'react-icons/pi';

export const NodeFieldElementSettings = memo(({ element }: { element: NodeFieldElement }) => {
  const { id, data } = element;
  const { showLabel, showDescription } = data;

  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const toggleShowLabel = useCallback(() => {
    dispatch(formElementNodeFieldDataChanged({ id, changes: { showLabel: !showLabel } }));
  }, [dispatch, id, showLabel]);

  const toggleShowDescription = useCallback(() => {
    dispatch(formElementNodeFieldDataChanged({ id, changes: { showDescription: !showDescription } }));
  }, [dispatch, id, showDescription]);

  return (
    <Popover placement="top">
      <PopoverTrigger>
        <IconButton aria-label="settings" icon={<PiWrenchFill />} variant="link" size="sm" alignSelf="stretch" />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverArrow />
        <PopoverBody>
          <FormControl>
            <FormLabel flex={1}>{t('workflows.builder.label')}</FormLabel>
            <Switch size="sm" isChecked={showLabel} onChange={toggleShowLabel} />
          </FormControl>
          <FormControl>
            <FormLabel flex={1}>{t('workflows.builder.description')}</FormLabel>
            <Switch size="sm" isChecked={showDescription} onChange={toggleShowDescription} />
          </FormControl>
          {data.config?.configType === 'integer-field-config' && (
            <NodeFieldElementIntegerConfig id={id} config={data.config} />
          )}
          {data.config?.configType === 'float-field-config' && (
            <NodeFieldElementFloatSettings id={id} config={data.config} />
          )}
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});
NodeFieldElementSettings.displayName = 'NodeFieldElementSettings';
