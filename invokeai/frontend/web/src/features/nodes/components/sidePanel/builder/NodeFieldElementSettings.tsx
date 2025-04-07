import {
  Flex,
  FormControl,
  FormLabel,
  IconButton,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Portal,
  Switch,
} from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { NodeFieldElementFloatSettings } from 'features/nodes/components/sidePanel/builder/NodeFieldElementFloatSettings';
import { NodeFieldElementIntegerSettings } from 'features/nodes/components/sidePanel/builder/NodeFieldElementIntegerSettings';
import { NodeFieldElementStringSettings } from 'features/nodes/components/sidePanel/builder/NodeFieldElementStringSettings';
import { useInputFieldTemplateOrThrow } from 'features/nodes/hooks/useInputFieldTemplateOrThrow';
import { formElementNodeFieldDataChanged } from 'features/nodes/store/nodesSlice';
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
  const fieldTemplate = useInputFieldTemplateOrThrow(nodeId, fieldName);

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
      <Portal>
        <PopoverContent>
          <PopoverArrow />
          <PopoverBody minW={48}>
            <Flex w="full" h="full" gap={2} flexDir="column">
              <FormControl>
                <FormLabel flex={1}>{t('workflows.builder.showDescription')}</FormLabel>
                <Switch size="sm" isChecked={showDescription} onChange={toggleShowDescription} />
              </FormControl>
              {settings?.type === 'integer-field-config' && isIntegerFieldInputTemplate(fieldTemplate) && (
                <NodeFieldElementIntegerSettings
                  id={id}
                  settings={settings}
                  nodeId={nodeId}
                  fieldName={fieldName}
                  fieldTemplate={fieldTemplate}
                />
              )}
              {settings?.type === 'float-field-config' && isFloatFieldInputTemplate(fieldTemplate) && (
                <NodeFieldElementFloatSettings
                  id={id}
                  settings={settings}
                  nodeId={nodeId}
                  fieldName={fieldName}
                  fieldTemplate={fieldTemplate}
                />
              )}
              {settings?.type === 'string-field-config' && (
                <NodeFieldElementStringSettings id={id} settings={settings} />
              )}
            </Flex>
          </PopoverBody>
        </PopoverContent>
      </Portal>
    </Popover>
  );
});
NodeFieldElementSettings.displayName = 'NodeFieldElementSettings';
