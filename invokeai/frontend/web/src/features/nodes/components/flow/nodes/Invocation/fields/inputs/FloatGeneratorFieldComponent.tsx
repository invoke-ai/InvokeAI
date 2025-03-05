import { Flex, Select, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { getOverlayScrollbarsParams, overlayScrollbarsStyles } from 'common/components/OverlayScrollbars/constants';
import { FloatGeneratorArithmeticSequenceSettings } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/FloatGeneratorArithmeticSequenceSettings';
import { FloatGeneratorLinearDistributionSettings } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/FloatGeneratorLinearDistributionSettings';
import { FloatGeneratorParseStringSettings } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/FloatGeneratorParseStringSettings';
import { FloatGeneratorUniformRandomDistributionSettings } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/FloatGeneratorUniformRandomDistributionSettings';
import type { FieldComponentProps } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/types';
import { fieldFloatGeneratorValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type { FloatGeneratorFieldInputInstance, FloatGeneratorFieldInputTemplate } from 'features/nodes/types/field';
import {
  FloatGeneratorArithmeticSequenceType,
  FloatGeneratorLinearDistributionType,
  FloatGeneratorParseStringType,
  FloatGeneratorUniformRandomDistributionType,
  getFloatGeneratorDefaults,
  resolveFloatGeneratorField,
} from 'features/nodes/types/field';
import { isNil, round } from 'lodash-es';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useDebounce } from 'use-debounce';

const overlayscrollbarsOptions = getOverlayScrollbarsParams().options;

export const FloatGeneratorFieldInputComponent = memo(
  (props: FieldComponentProps<FloatGeneratorFieldInputInstance, FloatGeneratorFieldInputTemplate>) => {
    const { nodeId, field } = props;
    const { t } = useTranslation();
    const dispatch = useAppDispatch();

    const onChange = useCallback(
      (value: FloatGeneratorFieldInputInstance['value']) => {
        dispatch(
          fieldFloatGeneratorValueChanged({
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
        const value = getFloatGeneratorDefaults(e.target.value as FloatGeneratorFieldInputInstance['value']['type']);
        if (!value) {
          return;
        }
        dispatch(
          fieldFloatGeneratorValueChanged({
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
        debouncedField.value.type === FloatGeneratorUniformRandomDistributionType &&
        isNil(debouncedField.value.seed)
      ) {
        const { count } = debouncedField.value;
        return `<${t('nodes.generatorNRandomValues', { count })}>`;
      }
      const resolvedValues = resolveFloatGeneratorField(debouncedField);
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
          <option value={FloatGeneratorArithmeticSequenceType}>{t('nodes.arithmeticSequence')}</option>
          <option value={FloatGeneratorLinearDistributionType}>{t('nodes.linearDistribution')}</option>
          <option value={FloatGeneratorUniformRandomDistributionType}>{t('nodes.uniformRandomDistribution')}</option>
          <option value={FloatGeneratorParseStringType}>{t('nodes.parseString')}</option>
        </Select>
        {field.value.type === FloatGeneratorArithmeticSequenceType && (
          <FloatGeneratorArithmeticSequenceSettings state={field.value} onChange={onChange} />
        )}
        {field.value.type === FloatGeneratorLinearDistributionType && (
          <FloatGeneratorLinearDistributionSettings state={field.value} onChange={onChange} />
        )}
        {field.value.type === FloatGeneratorUniformRandomDistributionType && (
          <FloatGeneratorUniformRandomDistributionSettings state={field.value} onChange={onChange} />
        )}
        {field.value.type === FloatGeneratorParseStringType && (
          <FloatGeneratorParseStringSettings state={field.value} onChange={onChange} />
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

FloatGeneratorFieldInputComponent.displayName = 'FloatGeneratorFieldInputComponent';
