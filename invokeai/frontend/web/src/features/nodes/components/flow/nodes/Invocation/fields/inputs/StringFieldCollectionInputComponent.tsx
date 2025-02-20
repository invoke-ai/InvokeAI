import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Button, Divider, Flex, FormLabel, Grid, GridItem, IconButton, Input } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/nanostores/store';
import { getOverlayScrollbarsParams, overlayScrollbarsStyles } from 'common/components/OverlayScrollbars/constants';
import { useInputFieldIsInvalid } from 'features/nodes/hooks/useInputFieldIsInvalid';
import { fieldStringCollectionValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type {
  StringFieldCollectionInputInstance,
  StringFieldCollectionInputTemplate,
} from 'features/nodes/types/field';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
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

export const StringFieldCollectionInputComponent = memo(
  (props: FieldComponentProps<StringFieldCollectionInputInstance, StringFieldCollectionInputTemplate>) => {
    const { nodeId, field } = props;
    const { t } = useTranslation();
    const store = useAppStore();

    const isInvalid = useInputFieldIsInvalid(nodeId, field.name);

    const onRemoveString = useCallback(
      (index: number) => {
        const newValue = field.value ? [...field.value] : [];
        newValue.splice(index, 1);
        store.dispatch(fieldStringCollectionValueChanged({ nodeId, fieldName: field.name, value: newValue }));
      },
      [field.name, field.value, nodeId, store]
    );

    const onChangeString = useCallback(
      (index: number, value: string) => {
        const newValue = field.value ? [...field.value] : [];
        newValue[index] = value;
        store.dispatch(fieldStringCollectionValueChanged({ nodeId, fieldName: field.name, value: newValue }));
      },
      [field.name, field.value, nodeId, store]
    );

    const onAddString = useCallback(() => {
      const newValue = field.value ? [...field.value, ''] : [''];
      store.dispatch(fieldStringCollectionValueChanged({ nodeId, fieldName: field.name, value: newValue }));
    }, [field.name, field.value, nodeId, store]);

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
        <Button onClick={onAddString} variant="ghost">
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
                  <ListItemContent
                    key={index}
                    value={value}
                    index={index}
                    onRemoveString={onRemoveString}
                    onChangeString={onChangeString}
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

StringFieldCollectionInputComponent.displayName = 'StringFieldCollectionInputComponent';

type StringListItemContentProps = {
  value: string;
  index: number;
  onRemoveString: (index: number) => void;
  onChangeString: (index: number, value: string) => void;
};

const StringListItemContent = memo(({ value, index, onRemoveString, onChangeString }: StringListItemContentProps) => {
  const { t } = useTranslation();

  const onClickRemove = useCallback(() => {
    onRemoveString(index);
  }, [index, onRemoveString]);
  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      onChangeString(index, e.target.value);
    },
    [index, onChangeString]
  );
  return (
    <Flex alignItems="center" gap={1}>
      <Input size="xs" resize="none" value={value} onChange={onChange} />
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
});
StringListItemContent.displayName = 'StringListItemContent';

type ListItemContentProps = {
  value: string;
  index: number;
  onRemoveString: (index: number) => void;
  onChangeString: (index: number, value: string) => void;
};

const ListItemContent = memo(({ value, index, onRemoveString, onChangeString }: ListItemContentProps) => {
  const { t } = useTranslation();

  const onClickRemove = useCallback(() => {
    onRemoveString(index);
  }, [index, onRemoveString]);
  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      onChangeString(index, e.target.value);
    },
    [index, onChangeString]
  );

  return (
    <>
      <GridItem>
        <FormLabel ps={1} m={0}>
          {index + 1}.
        </FormLabel>
      </GridItem>
      <GridItem>
        <Input size="sm" resize="none" value={value} onChange={onChange} />
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
});
ListItemContent.displayName = 'ListItemContent';
