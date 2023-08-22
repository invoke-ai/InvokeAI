import { useAppDispatch } from 'app/store/storeHooks';
import { fieldColorValueChanged } from 'features/nodes/store/nodesSlice';
import {
  ColorInputFieldTemplate,
  ColorInputFieldValue,
  FieldComponentProps,
} from 'features/nodes/types/types';
import { memo, useCallback } from 'react';
import { RgbaColor, RgbaColorPicker } from 'react-colorful';

const ColorInputFieldComponent = (
  props: FieldComponentProps<ColorInputFieldValue, ColorInputFieldTemplate>
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

export default memo(ColorInputFieldComponent);
