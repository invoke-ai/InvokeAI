import { Flex, Select, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { getOverlayScrollbarsParams, overlayScrollbarsStyles } from 'common/components/OverlayScrollbars/constants';
import { IntegerGeneratorRandomSettings } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/IntegerGeneratorRandomSettings';
import { IntegerGeneratorStartCountStepSettings } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/IntegerGeneratorStartCountStepSettings';
import { IntegerGeneratorStartEndStepSettings } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/IntegerGeneratorStartEndStepSettings';
import { fieldIntegerGeneratorValueChanged } from 'features/nodes/store/nodesSlice';
import {
  getIntegerGeneratorRandomDefaults,
  getIntegerGeneratorStartCountStepDefaults,
  getIntegerGeneratorStartEndStepDefaults,
  type IntegerGeneratorFieldInputInstance,
  type IntegerGeneratorFieldInputTemplate,
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
  if (generatorType === 'integer_generator_start_end_step') {
    return getIntegerGeneratorStartEndStepDefaults();
  }
  if (generatorType === 'integer_generator_start_count_step') {
    return getIntegerGeneratorStartCountStepDefaults();
  }
  if (generatorType === 'integer_generator_random') {
    return getIntegerGeneratorRandomDefaults();
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
          <option value="integer_generator_start_end_step">{t('nodes.startEndStep')}</option>
          <option value="integer_generator_start_count_step">{t('nodes.startCountStep')}</option>
          <option value="integer_generator_random">{t('nodes.random')}</option>
        </Select>
        {field.value.type === 'integer_generator_start_end_step' && (
          <IntegerGeneratorStartEndStepSettings state={field.value} onChange={onChange} />
        )}
        {field.value.type === 'integer_generator_start_count_step' && (
          <IntegerGeneratorStartCountStepSettings state={field.value} onChange={onChange} />
        )}
        {field.value.type === 'integer_generator_random' && (
          <IntegerGeneratorRandomSettings state={field.value} onChange={onChange} />
        )}
        {/* We don't show previews for random generators, bc they are non-deterministic */}
        {field.value.type !== 'integer_generator_random' && (
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
