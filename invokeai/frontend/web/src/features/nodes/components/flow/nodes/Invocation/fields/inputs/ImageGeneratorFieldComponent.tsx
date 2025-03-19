import { Flex, Select, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { getOverlayScrollbarsParams, overlayScrollbarsStyles } from 'common/components/OverlayScrollbars/constants';
import { ImageGeneratorImagesFromBoardSettings } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ImageGeneratorImagesFromBoardSettings';
import type { FieldComponentProps } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/types';
import { fieldImageGeneratorValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type { ImageGeneratorFieldInputInstance, ImageGeneratorFieldInputTemplate } from 'features/nodes/types/field';
import {
  getImageGeneratorDefaults,
  ImageGeneratorImagesFromBoardType,
  resolveImageGeneratorField,
} from 'features/nodes/types/field';
import { debounce } from 'lodash-es';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

const overlayscrollbarsOptions = getOverlayScrollbarsParams({}).options;

export const ImageGeneratorFieldInputComponent = memo(
  (props: FieldComponentProps<ImageGeneratorFieldInputInstance, ImageGeneratorFieldInputTemplate>) => {
    const { nodeId, field } = props;
    const { t } = useTranslation();
    const dispatch = useAppDispatch();

    const onChange = useCallback(
      (value: ImageGeneratorFieldInputInstance['value']) => {
        dispatch(
          fieldImageGeneratorValueChanged({
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
        const value = getImageGeneratorDefaults(e.target.value as ImageGeneratorFieldInputInstance['value']['type']);
        dispatch(
          fieldImageGeneratorValueChanged({
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
        debounce(async (field: ImageGeneratorFieldInputInstance) => {
          const resolvedValues = await resolveImageGeneratorField(field, dispatch);
          if (resolvedValues.length === 0) {
            setResolvedValuesAsString(`<${t('nodes.generatorNoValues')}>`);
          } else {
            setResolvedValuesAsString(`<${t('nodes.generatorImages', { count: resolvedValues.length })}>`);
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
          <option value={ImageGeneratorImagesFromBoardType}>{t('nodes.generatorImagesFromBoard')}</option>
        </Select>
        {field.value.type === ImageGeneratorImagesFromBoardType && (
          <ImageGeneratorImagesFromBoardSettings state={field.value} onChange={onChange} />
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

ImageGeneratorFieldInputComponent.displayName = 'ImageGeneratorFieldInputComponent';
