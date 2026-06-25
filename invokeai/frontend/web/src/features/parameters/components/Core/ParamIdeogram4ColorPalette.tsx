import { Flex, FormControl, FormLabel, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectIdeogram4ColorPalette, setIdeogram4ColorPalette } from 'features/controlLayers/store/paramsSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold, PiXBold } from 'react-icons/pi';

const MAX_COLORS = 16;
const DEFAULT_COLOR = '#808080';
const HEX_RE = /^#[0-9A-Fa-f]{6}$/;

const SWATCH_STYLE = {
  width: 28,
  height: 28,
  padding: 0,
  border: 'none',
  background: 'none',
  cursor: 'pointer',
} as const;

type ColorSwatchProps = {
  index: number;
  color: string;
  onSet: (index: number, value: string) => void;
  onRemove: (index: number) => void;
};

const ColorSwatch = memo(({ index, color, onSet, onRemove }: ColorSwatchProps) => {
  const { t } = useTranslation();
  const handleChange = useCallback((e: ChangeEvent<HTMLInputElement>) => onSet(index, e.target.value), [index, onSet]);
  const handleRemove = useCallback(() => onRemove(index), [index, onRemove]);
  return (
    <Flex alignItems="center" gap={1}>
      <input
        type="color"
        aria-label={t('parameters.ideogram4ColorPalette')}
        value={HEX_RE.test(color) ? color : DEFAULT_COLOR}
        onChange={handleChange}
        style={SWATCH_STYLE}
      />
      <IconButton
        aria-label={t('parameters.ideogram4RemoveColor')}
        icon={<PiXBold />}
        size="xs"
        variant="ghost"
        onClick={handleRemove}
      />
    </Flex>
  );
});
ColorSwatch.displayName = 'ColorSwatch';

// Up to 16 hex colors injected into the JSON caption's style_description.color_palette. Only applies
// in auto-build mode (ignored when the prompt is raw JSON).
const ParamIdeogram4ColorPalette = () => {
  const { t } = useTranslation();
  const palette = useAppSelector(selectIdeogram4ColorPalette);
  const dispatch = useAppDispatch();

  const setColor = useCallback(
    (index: number, value: string) => {
      const next = palette.slice();
      next[index] = value.toUpperCase();
      dispatch(setIdeogram4ColorPalette(next));
    },
    [palette, dispatch]
  );
  const removeColor = useCallback(
    (index: number) => {
      dispatch(setIdeogram4ColorPalette(palette.filter((_, i) => i !== index)));
    },
    [palette, dispatch]
  );
  const addColor = useCallback(() => {
    if (palette.length >= MAX_COLORS) {
      return;
    }
    dispatch(setIdeogram4ColorPalette([...palette, DEFAULT_COLOR]));
  }, [palette, dispatch]);

  return (
    <FormControl flexDir="column" alignItems="stretch" gap={2}>
      <FormLabel>{t('parameters.ideogram4ColorPalette')}</FormLabel>
      <Flex gap={2} flexWrap="wrap" alignItems="center">
        {palette.map((color, index) => (
          <ColorSwatch key={index} index={index} color={color} onSet={setColor} onRemove={removeColor} />
        ))}
        {palette.length < MAX_COLORS && (
          <IconButton
            aria-label={t('parameters.ideogram4AddColor')}
            icon={<PiPlusBold />}
            size="sm"
            variant="ghost"
            onClick={addColor}
          />
        )}
      </Flex>
    </FormControl>
  );
};

export default memo(ParamIdeogram4ColorPalette);
