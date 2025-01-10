import type { SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Box,
  CompositeNumberInput,
  Flex,
  FormControl,
  FormLabel,
  Grid,
  GridItem,
  IconButton,
  Text,
} from '@invoke-ai/ui-library';
import { NUMPY_RAND_MAX } from 'app/constants';
import { useAppStore } from 'app/store/nanostores/store';
import { useAppDispatch } from 'app/store/storeHooks';
import { getOverlayScrollbarsParams, overlayScrollbarsStyles } from 'common/components/OverlayScrollbars/constants';
import { useFieldIsInvalid } from 'features/nodes/hooks/useFieldIsInvalid';
import { fieldNumberCollectionValueChanged } from 'features/nodes/store/nodesSlice';
import type {
  FloatFieldCollectionInputInstance,
  FloatFieldCollectionInputTemplate,
  FloatStartStepCountGenerator,
  IntegerFieldCollectionInputInstance,
  IntegerFieldCollectionInputTemplate,
  IntegerStartStepCountGenerator,
} from 'features/nodes/types/field';
import { isNil } from 'lodash-es';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLightbulbFill, PiPencilSimpleFill, PiPlusBold, PiXBold } from 'react-icons/pi';

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
    const { nodeId, field, fieldTemplate } = props;
    const store = useAppStore();
    const isInvalid = useFieldIsInvalid(nodeId, field.name);
    const isIntegerField = useMemo(() => fieldTemplate.type.name === 'IntegerField', [fieldTemplate.type]);

    const entryMode = useMemo(() => {
      if (!field.value) {
        return 'manual';
      }
      if (Array.isArray(field.value)) {
        return 'manual';
      }
      return 'step';
    }, [field.value]);

    const toggleEntryMode = useCallback(() => {
      if (!field.value || Array.isArray(field.value)) {
        const newValue: IntegerStartStepCountGenerator | FloatStartStepCountGenerator = isIntegerField
          ? { type: 'integer-start-step-count-generator', start: 0, step: 1, count: 1 }
          : { type: 'float-start-step-count-generator', start: 0, step: 1, count: 1 };
        store.dispatch(fieldNumberCollectionValueChanged({ nodeId, fieldName: field.name, value: newValue }));
      } else {
        store.dispatch(
          fieldNumberCollectionValueChanged({
            nodeId,
            fieldName: field.name,
            value: [0],
          })
        );
      }
    }, [field.name, field.value, isIntegerField, nodeId, store]);

    const onAddNumber = useCallback(() => {
      const newValue = field.value && Array.isArray(field.value) ? [...field.value, 0] : [0];
      store.dispatch(fieldNumberCollectionValueChanged({ nodeId, fieldName: field.name, value: newValue }));
    }, [field.value, field.name, store, nodeId]);

    return (
      <Flex
        className="nodrag"
        position="relative"
        w="full"
        h="full"
        maxH={64}
        alignItems="stretch"
        justifyContent="center"
        flexDir="column"
        overflow="hidden"
        gap={1}
        p={1}
        borderWidth={1}
        borderRadius="base"
        sx={sx}
        data-error={isInvalid}
      >
        <Flex gap={2} w="full" alignItems="center">
          {!field.value ||
            (Array.isArray(field.value) && (
              <>
                <Text flexGrow={1}>Manual</Text>
                <IconButton
                  w="full"
                  onClick={onAddNumber}
                  aria-label="Add Item"
                  icon={<PiPlusBold />}
                  variant="ghost"
                  size="sm"
                />
              </>
            ))}
          {field.value && !Array.isArray(field.value) && (
            <>
              <Text flexGrow={1}>Generator</Text>
            </>
          )}
          <IconButton
            onClick={toggleEntryMode}
            aria-label="Toggle Entry Mode"
            icon={entryMode === 'manual' ? <PiLightbulbFill /> : <PiPencilSimpleFill />}
            variant="ghost"
            size="sm"
          />
        </Flex>
        {field.value && !Array.isArray(field.value) && (
          <GeneratorEntry nodeId={nodeId} fieldName={field.name} value={field.value} fieldTemplate={fieldTemplate} />
        )}
        {field.value && Array.isArray(field.value) && field.value.length > 0 && (
          <ManualEntry nodeId={nodeId} fieldName={field.name} value={field.value} fieldTemplate={fieldTemplate} />
        )}
      </Flex>
    );
  }
);

NumberFieldCollectionInputComponent.displayName = 'NumberFieldCollectionInputComponent';

const GeneratorEntry = ({
  nodeId,
  fieldName,
  value,
  fieldTemplate,
}: {
  nodeId: string;
  fieldName: string;
  value: IntegerStartStepCountGenerator | FloatStartStepCountGenerator;
  fieldTemplate: IntegerFieldCollectionInputTemplate | FloatFieldCollectionInputTemplate;
}) => {
  const dispatch = useAppDispatch();
  const isIntegerField = useMemo(() => fieldTemplate.type.name === 'IntegerField', [fieldTemplate.type]);
  const onChangeStart = useCallback(
    (v: number) => {
      const newValue = { ...value, start: v };
      dispatch(fieldNumberCollectionValueChanged({ nodeId, fieldName, value: newValue }));
    },
    [dispatch, fieldName, nodeId, value]
  );
  const onChangeCount = useCallback(
    (v: number) => {
      const newValue = { ...value, count: v };
      dispatch(fieldNumberCollectionValueChanged({ nodeId, fieldName, value: newValue }));
    },
    [dispatch, fieldName, nodeId, value]
  );
  const onChangeStep = useCallback(
    (v: number) => {
      const newValue = { ...value, step: v };
      dispatch(fieldNumberCollectionValueChanged({ nodeId, fieldName, value: newValue }));
    },
    [dispatch, fieldName, nodeId, value]
  );

  return (
    <Flex gap={2}>
      <FormControl>
        <FormLabel m={0}>Start</FormLabel>
        <CompositeNumberInput value={value.start} onChange={onChangeStart} min={-Infinity} max={Infinity} />
      </FormControl>
      <FormControl>
        <FormLabel m={0}>Count</FormLabel>
        <CompositeNumberInput value={value.count} onChange={onChangeCount} min={1} max={Infinity} />
      </FormControl>
      <FormControl>
        <FormLabel m={0}>Step</FormLabel>
        <CompositeNumberInput
          value={value.step}
          onChange={onChangeStep}
          min={-Infinity}
          max={Infinity}
          step={isIntegerField ? 1 : 0.1}
        />
      </FormControl>
    </Flex>
  );
};

const ManualEntry = ({
  nodeId,
  fieldName,
  value,
  fieldTemplate,
}: {
  nodeId: string;
  fieldName: string;
  value: number[];
  fieldTemplate: IntegerFieldCollectionInputTemplate | FloatFieldCollectionInputTemplate;
}) => {
  const dispatch = useAppDispatch();
  const isIntegerField = useMemo(() => fieldTemplate.type.name === 'IntegerField', [fieldTemplate.type]);

  const onRemoveNumber = useCallback(
    (index: number) => {
      const newValue = [...value];
      newValue.splice(index, 1);
      dispatch(
        fieldNumberCollectionValueChanged({
          nodeId,
          fieldName,
          value: newValue.length > 0 ? newValue : undefined,
        })
      );
    },
    [value, dispatch, nodeId, fieldName]
  );

  const onChangeNumber = useCallback(
    (index: number, num: number) => {
      const newValue = [...value];
      newValue[index] = num;
      dispatch(fieldNumberCollectionValueChanged({ nodeId, fieldName, value: newValue }));
    },
    [value, dispatch, nodeId, fieldName]
  );

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

  return (
    <Box w="full" h="full">
      <OverlayScrollbarsComponent
        className="nowheel"
        defer
        style={overlayScrollbarsStyles}
        options={overlayscrollbarsOptions}
      >
        <Grid w="full" h="full" templateColumns="repeat(1fr)" gap={1}>
          {value.map((value, index) => (
            <GridItem key={index} position="relative" className="nodrag">
              <NumberListItemContent
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
            </GridItem>
          ))}
        </Grid>
      </OverlayScrollbarsComponent>
    </Box>
  );
};

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
      <Flex alignItems="center" gap={1} w="full">
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
        <IconButton
          size="sm"
          variant="link"
          alignSelf="stretch"
          onClick={onClickRemove}
          icon={<PiXBold />}
          aria-label={t('common.remove')}
          tooltip={t('common.remove')}
        />
      </Flex>
    );
  }
);
NumberListItemContent.displayName = 'NumberListItemContent';
