import type { SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Button,
  CompositeNumberInput,
  Divider,
  Flex,
  FormLabel,
  Grid,
  GridItem,
  IconButton,
} from '@invoke-ai/ui-library';
import { NUMPY_RAND_MAX } from 'app/constants';
import { useAppStore } from 'app/store/nanostores/store';
import { getOverlayScrollbarsParams, overlayScrollbarsStyles } from 'common/components/OverlayScrollbars/constants';
import { useInputFieldIsInvalid } from 'features/nodes/hooks/useInputFieldIsInvalid';
import { fieldFloatCollectionValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type { FloatFieldCollectionInputInstance, FloatFieldCollectionInputTemplate } from 'features/nodes/types/field';
import { isNil } from 'lodash-es';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

import type { FieldComponentProps } from './types';

const overlayscrollbarsOptions = getOverlayScrollbarsParams().options;

const sx = {
  borderWidth: 1,
  '&[data-error=true]': {
    borderColor: 'error.500',
    borderStyle: 'solid',
  },
} satisfies SystemStyleObject;

export const FloatFieldCollectionInputComponent = memo(
  (props: FieldComponentProps<FloatFieldCollectionInputInstance, FloatFieldCollectionInputTemplate>) => {
    const { nodeId, field, fieldTemplate } = props;
    const store = useAppStore();
    const { t } = useTranslation();

    const isInvalid = useInputFieldIsInvalid(nodeId, field.name);

    const onChangeValue = useCallback(
      (value: FloatFieldCollectionInputInstance['value']) => {
        store.dispatch(fieldFloatCollectionValueChanged({ nodeId, fieldName: field.name, value }));
      },
      [field.name, nodeId, store]
    );
    const onRemoveNumber = useCallback(
      (index: number) => {
        const newValue = field.value ? [...field.value] : [];
        newValue.splice(index, 1);
        onChangeValue(newValue);
      },
      [field.value, onChangeValue]
    );

    const onChangeNumber = useCallback(
      (index: number, value: number) => {
        const newValue = field.value ? [...field.value] : [];
        newValue[index] = value;
        onChangeValue(newValue);
      },
      [field.value, onChangeValue]
    );

    const onAddNumber = useCallback(() => {
      const newValue = field.value ? [...field.value, 0] : [0];
      onChangeValue(newValue);
    }, [field.value, onChangeValue]);

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
        return 0.1;
      }
      return fieldTemplate.multipleOf;
    }, [fieldTemplate.multipleOf]);

    const fineStep = useMemo(() => {
      if (isNil(fieldTemplate.multipleOf)) {
        return 0.01;
      }
      return fieldTemplate.multipleOf;
    }, [fieldTemplate.multipleOf]);

    return (
      <Flex
        className={NO_DRAG_CLASS}
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
        <Button onClick={onAddNumber} variant="ghost">
          {t('nodes.addItem')}
        </Button>
        {field.value && field.value.length > 0 && (
          <>
            <Divider />
            <OverlayScrollbarsComponent
              className={NO_WHEEL_CLASS}
              defer
              style={overlayScrollbarsStyles}
              options={overlayscrollbarsOptions}
            >
              <Grid gap={1} gridTemplateColumns="auto 1fr auto" alignItems="center">
                {field.value.map((value, index) => (
                  <FloatListItemContent
                    key={index}
                    value={value}
                    index={index}
                    min={min}
                    max={max}
                    step={step}
                    fineStep={fineStep}
                    onRemoveNumber={onRemoveNumber}
                    onChangeNumber={onChangeNumber}
                  />
                ))}
              </Grid>
            </OverlayScrollbarsComponent>
          </>
        )}
      </Flex>
    );
  }
);

FloatFieldCollectionInputComponent.displayName = 'FloatFieldCollectionInputComponent';

type FloatListItemContentProps = {
  value: number;
  index: number;
  min: number;
  max: number;
  step: number;
  fineStep: number;
  onRemoveNumber: (index: number) => void;
  onChangeNumber: (index: number, value: number) => void;
};

const FloatListItemContent = memo(
  ({ value, index, min, max, step, fineStep, onRemoveNumber, onChangeNumber }: FloatListItemContentProps) => {
    const { t } = useTranslation();

    const onClickRemove = useCallback(() => {
      onRemoveNumber(index);
    }, [index, onRemoveNumber]);
    const onChange = useCallback(
      (value: number) => {
        onChangeNumber(index, value);
      },
      [index, onChangeNumber]
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
            className={NO_DRAG_CLASS}
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
FloatListItemContent.displayName = 'FloatListItemContent';
