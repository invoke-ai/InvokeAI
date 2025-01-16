import { Flex, Select, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { getOverlayScrollbarsParams, overlayScrollbarsStyles } from 'common/components/OverlayScrollbars/constants';
import { IntegerGeneratorArithmeticSequenceSettings } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/IntegerGeneratorArithmeticSequenceSettings';
import { IntegerGeneratorLinearDistributionSettings } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/IntegerGeneratorLinearDistributionSettings';
import { IntegerGeneratorUniformRandomDistributionSettings } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/IntegerGeneratorUniformRandomDistributionSettings';
import { fieldIntegerGeneratorValueChanged } from 'features/nodes/store/nodesSlice';
import type {
  IntegerGeneratorFieldInputInstance,
  IntegerGeneratorFieldInputTemplate,
} from 'features/nodes/types/field';
import {
  getIntegerGeneratorArithmeticSequenceDefaults,
  getIntegerGeneratorLinearDistributionDefaults,
  getIntegerGeneratorUniformRandomDistributionDefaults,
  IntegerGeneratorArithmeticSequenceType,
  IntegerGeneratorLinearDistributionType,
  IntegerGeneratorUniformRandomDistributionType,
  resolveIntegerGeneratorField,
} from 'features/nodes/types/field';
import { round } from 'lodash-es';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { FieldComponentProps } from './types';

const overlayscrollbarsOptions = getOverlayScrollbarsParams().options;

const getDefaultValue = (generatorType: string) => {
  if (generatorType === IntegerGeneratorArithmeticSequenceType) {
    return getIntegerGeneratorArithmeticSequenceDefaults();
  }
  if (generatorType === IntegerGeneratorLinearDistributionType) {
    return getIntegerGeneratorLinearDistributionDefaults();
  }
  if (generatorType === IntegerGeneratorUniformRandomDistributionType) {
    return getIntegerGeneratorUniformRandomDistributionDefaults();
  }
  return null;
};

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
        const value = getDefaultValue(e.target.value);
        if (!value) {
          return;
        }
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

    const resolvedValues = useMemo(() => resolveIntegerGeneratorField(field), [field]);
    const resolvedValuesAsString = useMemo(() => {
      if (resolvedValues.length === 0) {
        return '<empty>';
      } else {
        return resolvedValues.map((val) => round(val, 2)).join(', ');
      }
    }, [resolvedValues]);

    return (
      <Flex flexDir="column" gap={2}>
        <Select className="nowheel nodrag" onChange={onChangeGeneratorType} value={field.value.type} size="sm">
          <option value={IntegerGeneratorArithmeticSequenceType}>{t('nodes.arithmeticSequence')}</option>
          <option value={IntegerGeneratorLinearDistributionType}>{t('nodes.linearDistribution')}</option>
          <option value={IntegerGeneratorUniformRandomDistributionType}>{t('nodes.uniformRandomDistribution')}</option>
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
        {/* We don't show previews for random generators, bc they are non-deterministic */}
        {field.value.type !== IntegerGeneratorUniformRandomDistributionType && (
          <Flex w="full" h="full" p={2} borderWidth={1} borderRadius="base" maxH={128}>
            <Flex w="full" h="auto">
              <OverlayScrollbarsComponent
                className="nodrag nowheel"
                defer
                style={overlayScrollbarsStyles}
                options={overlayscrollbarsOptions}
              >
                <Text className="nodrag nowheel" fontFamily="monospace" userSelect="text" cursor="text">
                  {resolvedValuesAsString}
                </Text>
              </OverlayScrollbarsComponent>
            </Flex>
          </Flex>
        )}
      </Flex>
    );
  }
);

IntegerGeneratorFieldInputComponent.displayName = 'IntegerGeneratorFieldInputComponent';
