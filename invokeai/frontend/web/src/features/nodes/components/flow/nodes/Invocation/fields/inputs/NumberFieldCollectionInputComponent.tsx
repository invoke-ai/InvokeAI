import type { SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Button,
  CompositeNumberInput,
  Divider,
  Flex,
  FormControl,
  FormLabel,
  Grid,
  GridItem,
  IconButton,
  Switch,
  Text,
} from '@invoke-ai/ui-library';
import { NUMPY_RAND_MAX } from 'app/constants';
import { useAppStore } from 'app/store/nanostores/store';
import { getOverlayScrollbarsParams, overlayScrollbarsStyles } from 'common/components/OverlayScrollbars/constants';
import { FloatRangeGenerator } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/FloatRangeGenerator';
import { useFieldIsInvalid } from 'features/nodes/hooks/useFieldIsInvalid';
import {
  fieldNumberCollectionGeneratorCommitted,
  fieldNumberCollectionGeneratorStateChanged,
  fieldNumberCollectionGeneratorToggled,
  fieldNumberCollectionLockLinearViewToggled,
  fieldNumberCollectionValueChanged,
} from 'features/nodes/store/nodesSlice';
import type {
  FloatFieldCollectionInputInstance,
  FloatFieldCollectionInputTemplate,
  IntegerFieldCollectionInputInstance,
  IntegerFieldCollectionInputTemplate,
} from 'features/nodes/types/field';
import { resolveNumberFieldCollectionValue } from 'features/nodes/types/fieldValidators';
import type {
  FloatRangeStartStepCountGenerator,
  IntegerRangeStartStepCountGenerator,
} from 'features/nodes/types/generators';
import { isNil, round } from 'lodash-es';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLockSimpleFill, PiLockSimpleOpenFill, PiXBold } from 'react-icons/pi';

import type { FieldComponentProps } from './types';

const overlayscrollbarsOptions = getOverlayScrollbarsParams().options;

const sx = {
  borderWidth: 1,
  '&[data-error=true]': {
    borderColor: 'error.500',
    borderStyle: 'solid',
  },
} satisfies SystemStyleObject;

export const NumberFieldCollectionInputComponent = memo(
  (
    props:
      | FieldComponentProps<IntegerFieldCollectionInputInstance, IntegerFieldCollectionInputTemplate>
      | FieldComponentProps<FloatFieldCollectionInputInstance, FloatFieldCollectionInputTemplate>
  ) => {
    const { nodeId, field, fieldTemplate, isLinearView } = props;
    const store = useAppStore();
    const { t } = useTranslation();

    const isInvalid = useFieldIsInvalid(nodeId, field.name);
    const isIntegerField = useMemo(() => fieldTemplate.type.name === 'IntegerField', [fieldTemplate.type]);

    const onRemoveNumber = useCallback(
      (index: number) => {
        const newValue = field.value ? [...field.value] : [];
        newValue.splice(index, 1);
        store.dispatch(fieldNumberCollectionValueChanged({ nodeId, fieldName: field.name, value: newValue }));
      },
      [field.name, field.value, nodeId, store]
    );

    const onChangeNumber = useCallback(
      (index: number, value: number) => {
        const newValue = field.value ? [...field.value] : [];
        newValue[index] = value;
        store.dispatch(fieldNumberCollectionValueChanged({ nodeId, fieldName: field.name, value: newValue }));
      },
      [field.name, field.value, nodeId, store]
    );

    const onAddNumber = useCallback(() => {
      const newValue = field.value ? [...field.value, 0] : [0];
      store.dispatch(fieldNumberCollectionValueChanged({ nodeId, fieldName: field.name, value: newValue }));
    }, [field.name, field.value, nodeId, store]);

    const min = useMemo(() => {
      let min = -NUMPY_RAND_MAX;
      if (!isNil(fieldTemplate.minimum)) {
        min = fieldTemplate.minimum;
      }
      if (!isNil(fieldTemplate.exclusiveMinimum)) {
        min = fieldTemplate.exclusiveMinimum + 0.01;
      }
      return min;
    }, [fieldTemplate.exclusiveMinimum, fieldTemplate.minimum]);

    const max = useMemo(() => {
      let max = NUMPY_RAND_MAX;
      if (!isNil(fieldTemplate.maximum)) {
        max = fieldTemplate.maximum;
      }
      if (!isNil(fieldTemplate.exclusiveMaximum)) {
        max = fieldTemplate.exclusiveMaximum - 0.01;
      }
      return max;
    }, [fieldTemplate.exclusiveMaximum, fieldTemplate.maximum]);

    const step = useMemo(() => {
      if (isNil(fieldTemplate.multipleOf)) {
        return isIntegerField ? 1 : 0.1;
      }
      return fieldTemplate.multipleOf;
    }, [fieldTemplate.multipleOf, isIntegerField]);

    const fineStep = useMemo(() => {
      if (isNil(fieldTemplate.multipleOf)) {
        return isIntegerField ? 1 : 0.01;
      }
      return fieldTemplate.multipleOf;
    }, [fieldTemplate.multipleOf, isIntegerField]);

    const toggleGenerator = useCallback(() => {
      store.dispatch(fieldNumberCollectionGeneratorToggled({ nodeId, fieldName: field.name }));
    }, [field.name, nodeId, store]);

    const onChangeGenerator = useCallback(
      (generatorState: FloatRangeStartStepCountGenerator | IntegerRangeStartStepCountGenerator) => {
        store.dispatch(fieldNumberCollectionGeneratorStateChanged({ nodeId, fieldName: field.name, generatorState }));
      },
      [field.name, nodeId, store]
    );

    const onCommitGenerator = useCallback(() => {
      store.dispatch(fieldNumberCollectionGeneratorCommitted({ nodeId, fieldName: field.name }));
    }, [field.name, nodeId, store]);

    const onToggleLockLinearView = useCallback(() => {
      store.dispatch(fieldNumberCollectionLockLinearViewToggled({ nodeId, fieldName: field.name }));
    }, [field.name, nodeId, store]);

    const valuesAsString = useMemo(() => {
      const resolvedValue = resolveNumberFieldCollectionValue(field);
      return resolvedValue ? resolvedValue.map((val) => round(val, 2)).join(', ') : '';
    }, [field]);

    const isLockedOnLinearView = !(field.lockLinearView && isLinearView);

    return (
      <Flex
        className="nodrag"
        position="relative"
        w="full"
        h="auto"
        maxH={64}
        alignItems="stretch"
        justifyContent="center"
        p={1}
        sx={sx}
        data-error={isInvalid}
        borderRadius="base"
        flexDir="column"
        gap={1}
      >
        <Flex w="full" gap={2}>
          {!field.generator && (
            <Button onClick={onAddNumber} variant="ghost" flexGrow={1} size="sm">
              {t('nodes.addValue')}
            </Button>
          )}
          {field.generator && isLockedOnLinearView && (
            <Button
              tooltip={
                <Flex p={1} flexDir="column">
                  <Text fontWeight="semibold">{t('nodes.generatedValues')}:</Text>
                  <Text fontFamily="monospace">{valuesAsString}</Text>
                </Flex>
              }
              onClick={onCommitGenerator}
              variant="ghost"
              flexGrow={1}
              size="sm"
            >
              {t('nodes.commitValues')}
            </Button>
          )}
          {isLockedOnLinearView && (
            <FormControl w="min-content" pe={isLinearView ? 2 : undefined}>
              <FormLabel m={0}>{t('nodes.generator')}</FormLabel>
              <Switch onChange={toggleGenerator} isChecked={Boolean(field.generator)} size="sm" />
            </FormControl>
          )}
          {!isLinearView && (
            <IconButton
              onClick={onToggleLockLinearView}
              tooltip={field.lockLinearView ? t('nodes.unlockLinearView') : t('nodes.lockLinearView')}
              aria-label={field.lockLinearView ? t('nodes.unlockLinearView') : t('nodes.lockLinearView')}
              icon={field.lockLinearView ? <PiLockSimpleFill /> : <PiLockSimpleOpenFill />}
              variant="ghost"
              size="sm"
            />
          )}
        </Flex>
        {!field.generator && field.value && field.value.length > 0 && (
          <>
            {!(field.lockLinearView && isLinearView) && <Divider />}
            <OverlayScrollbarsComponent
              className="nowheel"
              defer
              style={overlayScrollbarsStyles}
              options={overlayscrollbarsOptions}
            >
              <Grid gap={1} gridTemplateColumns="auto 1fr auto" alignItems="center">
                {field.value.map((value, index) => (
                  <NumberListItemContent
                    key={index}
                    value={value}
                    index={index}
                    min={min}
                    max={max}
                    step={step}
                    fineStep={fineStep}
                    isIntegerField={isIntegerField}
                    onRemoveNumber={onRemoveNumber}
                    onChangeNumber={onChangeNumber}
                  />
                ))}
              </Grid>
            </OverlayScrollbarsComponent>
          </>
        )}
        {field.generator && field.generator.type === 'float-range-generator-start-step-count' && (
          <>
            {!(field.lockLinearView && isLinearView) && <Divider />}
            <FloatRangeGenerator state={field.generator} onChange={onChangeGenerator} />
          </>
        )}
      </Flex>
    );
  }
);

NumberFieldCollectionInputComponent.displayName = 'NumberFieldCollectionInputComponent';

type NumberListItemContentProps = {
  value: number;
  index: number;
  isIntegerField: boolean;
  min: number;
  max: number;
  step: number;
  fineStep: number;
  onRemoveNumber: (index: number) => void;
  onChangeNumber: (index: number, value: number) => void;
};

const NumberListItemContent = memo(
  ({
    value,
    index,
    isIntegerField,
    min,
    max,
    step,
    fineStep,
    onRemoveNumber,
    onChangeNumber,
  }: NumberListItemContentProps) => {
    const { t } = useTranslation();

    const onClickRemove = useCallback(() => {
      onRemoveNumber(index);
    }, [index, onRemoveNumber]);
    const onChange = useCallback(
      (v: number) => {
        onChangeNumber(index, isIntegerField ? Math.floor(Number(v)) : Number(v));
      },
      [index, isIntegerField, onChangeNumber]
    );

    return (
      <>
        <GridItem>
          <FormLabel ps={1} m={0}>
            {index + 1}.
          </FormLabel>
        </GridItem>
        <GridItem>
          <CompositeNumberInput
            onChange={onChange}
            value={value}
            min={min}
            max={max}
            step={step}
            fineStep={fineStep}
            className="nodrag"
            flexGrow={1}
          />
        </GridItem>
        <GridItem>
          <IconButton
            tabIndex={-1}
            size="sm"
            variant="link"
            alignSelf="stretch"
            onClick={onClickRemove}
            icon={<PiXBold />}
            aria-label={t('common.delete')}
          />
        </GridItem>
      </>
    );
  }
);
NumberListItemContent.displayName = 'NumberListItemContent';
