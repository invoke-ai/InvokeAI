import { useAppDispatch } from 'app/store/storeHooks';
import { fieldColorValueChanged } from 'features/nodes/store/nodesSlice';
import {
  ColorFieldInputTemplate,
  ColorFieldInputInstance,
} from 'features/nodes/types/field';
import { FieldComponentProps } from './types';
import { memo, useCallback } from 'react';
import { RgbaColor, RgbaColorPicker } from 'react-colorful';

const ColorFieldInputComponent = (
  props: FieldComponentProps<ColorFieldInputInstance, ColorFieldInputTemplate>
) => {
  const { nodeId, field } = props;

  const dispatch = useAppDispatch();

  const handleValueChanged = useCallback(
    (value: RgbaColor) => {
      dispatch(
        fieldColorValueChanged({
          nodeId,
          fieldName: field.name,
          value,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  return (
    <RgbaColorPicker
      className="nodrag"
      color={field.value}
      onChange={handleValueChanged}
    />
  );
};

export default memo(ColorFieldInputComponent);
