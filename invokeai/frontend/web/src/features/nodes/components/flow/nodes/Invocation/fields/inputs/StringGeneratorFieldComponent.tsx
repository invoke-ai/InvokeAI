import { Flex, Select, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { getOverlayScrollbarsParams, overlayScrollbarsStyles } from 'common/components/OverlayScrollbars/constants';
import { StringGeneratorDynamicPromptsCombinatorialSettings } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/StringGeneratorDynamicPromptsCombinatorialSettings';
import { StringGeneratorDynamicPromptsRandomSettings } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/StringGeneratorDynamicPromptsRandomSettings';
import { StringGeneratorParseStringSettings } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/StringGeneratorParseStringSettings';
import type { FieldComponentProps } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/types';
import { fieldStringGeneratorValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type { StringGeneratorFieldInputInstance, StringGeneratorFieldInputTemplate } from 'features/nodes/types/field';
import {
  getStringGeneratorDefaults,
  resolveStringGeneratorField,
  StringGeneratorDynamicPromptsCombinatorialType,
  StringGeneratorDynamicPromptsRandomType,
  StringGeneratorParseStringType,
} from 'features/nodes/types/field';
import { debounce } from 'lodash-es';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

const overlayscrollbarsOptions = getOverlayScrollbarsParams().options;

export const StringGeneratorFieldInputComponent = memo(
  (props: FieldComponentProps<StringGeneratorFieldInputInstance, StringGeneratorFieldInputTemplate>) => {
    const { nodeId, field } = props;
    const { t } = useTranslation();
    const dispatch = useAppDispatch();

    const onChange = useCallback(
      (value: StringGeneratorFieldInputInstance['value']) => {
        dispatch(
          fieldStringGeneratorValueChanged({
            nodeId,
            fieldName: field.name,
            value,
          })
        );
      },
      [dispatch, field.name, nodeId]
    );

    const onChangeGeneratorType = useCallback(
      (e: ChangeEvent<HTMLSelectElement>) => {
        const value = getStringGeneratorDefaults(e.target.value as StringGeneratorFieldInputInstance['value']['type']);
        dispatch(
          fieldStringGeneratorValueChanged({
            nodeId,
            fieldName: field.name,
            value,
          })
        );
      },
      [dispatch, field.name, nodeId]
    );

    const [resolvedValuesAsString, setResolvedValuesAsString] = useState<string | null>(null);
    const resolveAndSetValuesAsString = useMemo(
      () =>
        debounce(async (field: StringGeneratorFieldInputInstance) => {
          const resolvedValues = await resolveStringGeneratorField(field, dispatch);
          if (resolvedValues.length === 0) {
            setResolvedValuesAsString(`<${t('nodes.generatorNoValues')}>`);
          } else {
            setResolvedValuesAsString(resolvedValues.join(', '));
          }
        }, 300),
      [dispatch, t]
    );
    useEffect(() => {
      resolveAndSetValuesAsString(field);
    }, [field, resolveAndSetValuesAsString]);

    return (
      <Flex flexDir="column" gap={2} flexGrow={1}>
        <Select
          className={`${NO_WHEEL_CLASS} ${NO_DRAG_CLASS}`}
          onChange={onChangeGeneratorType}
          value={field.value.type}
          size="sm"
        >
          <option value={StringGeneratorParseStringType}>{t('nodes.parseString')}</option>
          <option value={StringGeneratorDynamicPromptsRandomType}>{t('nodes.dynamicPromptsRandom')}</option>
          <option value={StringGeneratorDynamicPromptsCombinatorialType}>
            {t('nodes.dynamicPromptsCombinatorial')}
          </option>
        </Select>
        {field.value.type === StringGeneratorParseStringType && (
          <StringGeneratorParseStringSettings state={field.value} onChange={onChange} />
        )}
        {field.value.type === StringGeneratorDynamicPromptsRandomType && (
          <StringGeneratorDynamicPromptsRandomSettings state={field.value} onChange={onChange} />
        )}
        {field.value.type === StringGeneratorDynamicPromptsCombinatorialType && (
          <StringGeneratorDynamicPromptsCombinatorialSettings state={field.value} onChange={onChange} />
        )}
        <Flex w="full" h="full" p={2} borderWidth={1} borderRadius="base" maxH={128}>
          <Flex w="full" h="auto">
            <OverlayScrollbarsComponent
              className={`${NO_WHEEL_CLASS} ${NO_DRAG_CLASS}`}
              defer
              style={overlayScrollbarsStyles}
              options={overlayscrollbarsOptions}
            >
              <Text
                className={`${NO_WHEEL_CLASS} ${NO_DRAG_CLASS}`}
                fontFamily="monospace"
                userSelect="text"
                cursor="text"
              >
                {resolvedValuesAsString}
              </Text>
            </OverlayScrollbarsComponent>
          </Flex>
        </Flex>
      </Flex>
    );
  }
);

StringGeneratorFieldInputComponent.displayName = 'StringGeneratorFieldInputComponent';
