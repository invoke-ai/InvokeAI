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
import { NodeFieldElementStringSettings } from 'features/nodes/components/sidePanel/builder/NodeFieldElementStringSettings';
import { useInputFieldTemplate } from 'features/nodes/hooks/useInputFieldTemplate';
import { formElementNodeFieldDataChanged } from 'features/nodes/store/workflowSlice';
import {
  isFloatFieldInputTemplate,
  isIntegerFieldInputTemplate,
  isStringFieldInputTemplate,
} from 'features/nodes/types/field';
import {
  getFloatFieldSettingsDefaults,
  getIntegerFieldSettingsDefaults,
  getStringFieldSettingsDefaults,
  type NodeFieldElement,
} from 'features/nodes/types/workflow';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiWrenchFill } from 'react-icons/pi';

export const NodeFieldElementSettings = memo(({ element }: { element: NodeFieldElement }) => {
  const { id, data } = element;
  const { showDescription, fieldIdentifier } = data;
  const { nodeId, fieldName } = fieldIdentifier;
  const fieldTemplate = useInputFieldTemplate(nodeId, fieldName);

  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const toggleShowDescription = useCallback(() => {
    dispatch(formElementNodeFieldDataChanged({ id, changes: { showDescription: !showDescription } }));
  }, [dispatch, id, showDescription]);

  // If settings are not present, set defaults based on field type.
  const settings = useMemo(() => {
    if (data.settings) {
      return data.settings;
    }
    if (isIntegerFieldInputTemplate(fieldTemplate)) {
      return getIntegerFieldSettingsDefaults();
    }
    if (isFloatFieldInputTemplate(fieldTemplate)) {
      return getFloatFieldSettingsDefaults();
    }
    if (isStringFieldInputTemplate(fieldTemplate)) {
      return getStringFieldSettingsDefaults();
    }
    return null;
  }, [data.settings, fieldTemplate]);

  return (
    <Popover placement="top" isLazy lazyBehavior="unmount">
      <PopoverTrigger>
        <IconButton aria-label="settings" icon={<PiWrenchFill />} variant="link" size="sm" alignSelf="stretch" />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverArrow />
        <PopoverBody minW={48}>
          <FormControl>
            <FormLabel flex={1}>{t('workflows.builder.showDescription')}</FormLabel>
            <Switch size="sm" isChecked={showDescription} onChange={toggleShowDescription} />
          </FormControl>
          {settings?.type === 'integer-field-config' && <NodeFieldElementIntegerConfig id={id} config={settings} />}
          {settings?.type === 'float-field-config' && <NodeFieldElementFloatSettings id={id} config={settings} />}
          {settings?.type === 'string-field-config' && <NodeFieldElementStringSettings id={id} config={settings} />}
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});
NodeFieldElementSettings.displayName = 'NodeFieldElementSettings';
