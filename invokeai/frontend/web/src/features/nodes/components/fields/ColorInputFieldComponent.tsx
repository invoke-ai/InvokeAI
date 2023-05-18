import {
  ColorInputFieldTemplate,
  ColorInputFieldValue,
} from 'features/nodes/types/types';
import { memo } from 'react';
import { FieldComponentProps } from './types';
import { RgbaColor, RgbaColorPicker } from 'react-colorful';
import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import { useAppDispatch } from 'app/store/storeHooks';

const ColorInputFieldComponent = (
  props: FieldComponentProps<ColorInputFieldValue, ColorInputFieldTemplate>
) => {
  const { nodeId, field } = props;

  const dispatch = useAppDispatch();

  const handleValueChanged = (value: RgbaColor) => {
    dispatch(fieldValueChanged({ nodeId, fieldName: field.name, value }));
  };

  return (
    <RgbaColorPicker
      className="nodrag"
      color={field.value}
      onChange={handleValueChanged}
    />
  );
};

export default memo(ColorInputFieldComponent);
