import { Flex, Select, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { getOverlayScrollbarsParams, overlayScrollbarsStyles } from 'common/components/OverlayScrollbars/constants';
import { IntegerGeneratorArithmeticSequenceSettings } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/IntegerGeneratorArithmeticSequenceSettings';
import { IntegerGeneratorLinearDistributionSettings } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/IntegerGeneratorLinearDistributionSettings';
import { IntegerGeneratorParseStringSettings } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/IntegerGeneratorParseStringSettings';
import { IntegerGeneratorUniformRandomDistributionSettings } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/IntegerGeneratorUniformRandomDistributionSettings';
import type { FieldComponentProps } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/types';
import { fieldIntegerGeneratorValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type {
  IntegerGeneratorFieldInputInstance,
  IntegerGeneratorFieldInputTemplate,
} from 'features/nodes/types/field';
import {
  getIntegerGeneratorDefaults,
  IntegerGeneratorArithmeticSequenceType,
  IntegerGeneratorLinearDistributionType,
  IntegerGeneratorParseStringType,
  IntegerGeneratorUniformRandomDistributionType,
  resolveIntegerGeneratorField,
} from 'features/nodes/types/field';
import { isNil, round } from 'lodash-es';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useDebounce } from 'use-debounce';

const overlayscrollbarsOptions = getOverlayScrollbarsParams().options;

export const IntegerGeneratorFieldInputComponent = memo(
  (props: FieldComponentProps<IntegerGeneratorFieldInputInstance, IntegerGeneratorFieldInputTemplate>) => {
    const { nodeId, field } = props;
    const { t } = useTranslation();
    const dispatch = useAppDispatch();

    const onChange = useCallback(
      (value: IntegerGeneratorFieldInputInstance['value']) => {
        dispatch(
          fieldIntegerGeneratorValueChanged({
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
        const value = getIntegerGeneratorDefaults(
          e.target.value as IntegerGeneratorFieldInputInstance['value']['type']
        );
        dispatch(
          fieldIntegerGeneratorValueChanged({
            nodeId,
            fieldName: field.name,
            value,
          })
        );
      },
      [dispatch, field.name, nodeId]
    );

    const [debouncedField] = useDebounce(field, 300);
    const resolvedValuesAsString = useMemo(() => {
      if (
        debouncedField.value.type === IntegerGeneratorUniformRandomDistributionType &&
        isNil(debouncedField.value.seed)
      ) {
        const { count } = debouncedField.value;
        return `<${t('nodes.generatorNRandomValues', { count })}>`;
      }
      const resolvedValues = resolveIntegerGeneratorField(debouncedField);
      if (resolvedValues.length === 0) {
        return `<${t('nodes.generatorNoValues')}>`;
      } else {
        return resolvedValues.map((val) => round(val, 2)).join(', ');
      }
    }, [debouncedField, t]);

    return (
      <Flex flexDir="column" gap={2}>
        <Select
          className={`${NO_WHEEL_CLASS} ${NO_DRAG_CLASS}`}
          onChange={onChangeGeneratorType}
          value={field.value.type}
          size="sm"
        >
          <option value={IntegerGeneratorArithmeticSequenceType}>{t('nodes.arithmeticSequence')}</option>
          <option value={IntegerGeneratorLinearDistributionType}>{t('nodes.linearDistribution')}</option>
          <option value={IntegerGeneratorUniformRandomDistributionType}>{t('nodes.uniformRandomDistribution')}</option>
          <option value={IntegerGeneratorParseStringType}>{t('nodes.parseString')}</option>
        </Select>
        {field.value.type === IntegerGeneratorArithmeticSequenceType && (
          <IntegerGeneratorArithmeticSequenceSettings state={field.value} onChange={onChange} />
        )}
        {field.value.type === IntegerGeneratorLinearDistributionType && (
          <IntegerGeneratorLinearDistributionSettings state={field.value} onChange={onChange} />
        )}
        {field.value.type === IntegerGeneratorUniformRandomDistributionType && (
          <IntegerGeneratorUniformRandomDistributionSettings state={field.value} onChange={onChange} />
        )}
        {field.value.type === IntegerGeneratorParseStringType && (
          <IntegerGeneratorParseStringSettings state={field.value} onChange={onChange} />
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

IntegerGeneratorFieldInputComponent.displayName = 'IntegerGeneratorFieldInputComponent';
