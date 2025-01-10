import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, Grid, GridItem, IconButton, Textarea } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/nanostores/store';
import { getOverlayScrollbarsParams, overlayScrollbarsStyles } from 'common/components/OverlayScrollbars/constants';
import { useFieldIsInvalid } from 'features/nodes/hooks/useFieldIsInvalid';
import { fieldStringCollectionValueChanged } from 'features/nodes/store/nodesSlice';
import type {
  StringFieldCollectionInputInstance,
  StringFieldCollectionInputTemplate,
} from 'features/nodes/types/field';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold, PiXBold } from 'react-icons/pi';

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
    const store = useAppStore();

    const isInvalid = useFieldIsInvalid(nodeId, field.name);

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
        className="nodrag"
        position="relative"
        w="full"
        h="full"
        maxH={64}
        alignItems="stretch"
        justifyContent="center"
      >
        {(!field.value || field.value.length === 0) && (
          <Box w="full" sx={sx} data-error={isInvalid} borderRadius="base">
            <IconButton
              w="full"
              onClick={onAddString}
              aria-label="Add Item"
              icon={<PiPlusBold />}
              variant="ghost"
              size="sm"
            />
          </Box>
        )}
        {field.value && field.value.length > 0 && (
          <Box w="full" h="auto" p={1} sx={sx} data-error={isInvalid} borderRadius="base">
            <OverlayScrollbarsComponent
              className="nowheel"
              defer
              style={overlayScrollbarsStyles}
              options={overlayscrollbarsOptions}
            >
              <Grid w="full" h="full" templateColumns="repeat(1, 1fr)" gap={1}>
                <IconButton
                  onClick={onAddString}
                  aria-label="Add Item"
                  icon={<PiPlusBold />}
                  variant="ghost"
                  size="sm"
                />
                {field.value.map((value, index) => (
                  <GridItem key={index} position="relative" className="nodrag">
                    <StringListItemContent
                      value={value}
                      index={index}
                      onRemoveString={onRemoveString}
                      onChangeString={onChangeString}
                    />
                  </GridItem>
                ))}
              </Grid>
            </OverlayScrollbarsComponent>
          </Box>
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
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      onChangeString(index, e.target.value);
    },
    [index, onChangeString]
  );
  return (
    <Flex alignItems="center" gap={1}>
      <Textarea size="xs" resize="none" value={value} onChange={onChange} />
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
