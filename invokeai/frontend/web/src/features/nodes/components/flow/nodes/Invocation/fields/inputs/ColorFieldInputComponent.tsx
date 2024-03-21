import { Box } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { hexToRGBA, rgbaToHex } from 'common/util/colorCodeTransformers';
import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { fieldColorValueChanged } from 'features/nodes/store/nodesSlice';
import type { ColorFieldInputInstance, ColorFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback, useMemo } from 'react';
import type { RgbaColor } from 'react-colorful';
import { HexColorInput, RgbaColorPicker } from 'react-colorful';

import type { FieldComponentProps } from './types';

const FALLBACK_COLOR: RgbaColor = { r: 0, g: 0, b: 0, a: 255 };

const ColorFieldInputComponent = (props: FieldComponentProps<ColorFieldInputInstance, ColorFieldInputTemplate>) => {
  const { nodeId, field } = props;

  const dispatch = useAppDispatch();

  const color = useMemo(() => {
    // For better or worse, zColorFieldValue is typed as optional. This means that `field.value` and `fieldTemplate.default`
    // can be undefined. Rather than changing the schema (which could have other consequences), we can just provide a fallback.
    if (!field.value) {
      return FALLBACK_COLOR;
    }
    const { r, g, b, a } = field.value;
    // We need to divide by 255 to convert from 0-255 to 0-1, which is what the UI component needs
    return { r, g, b, a: a / 255 };
  }, [field.value]);

  const handleValueChanged = useCallback(
    (value: RgbaColor | string) => {
      // We need to multiply by 255 to convert from 0-1 to 0-255, which is what the backend needs
      if (typeof value === 'string') {
        value = hexToRGBA(value, 1);
      }

      const { r, g, b, a: _a } = value;
      const a = Math.round(_a * 255);
      dispatch(
        fieldColorValueChanged({
          nodeId,
          fieldName: field.name,
          value: { r, g, b, a },
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <HexColorInput
        style={{
          background: colorTokenToCssVar('base.700'),
          color: colorTokenToCssVar('base.100'),
          fontSize: 12,
          paddingInlineStart: 10,
          borderRadius: 4,
          paddingBlock: 4,
          outline: 'none',
        }}
        className="nodrag"
        color={rgbaToHex(color, true)}
        onChange={handleValueChanged}
        prefixed
        alpha
      />
      <RgbaColorPicker className="nodrag" color={color} onChange={handleValueChanged} style={{ width: '100%' }} />
    </Box>
  );
};

export default memo(ColorFieldInputComponent);
