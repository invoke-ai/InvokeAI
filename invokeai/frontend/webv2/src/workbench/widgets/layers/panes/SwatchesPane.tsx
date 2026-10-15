import type { KeyboardEvent as ReactKeyboardEvent, MouseEvent as ReactMouseEvent, ReactNode } from 'react';

import { chakra, Flex, HStack, Separator, Stack, Text } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { formatHexColor, normalizeHex, parseHexColor } from '@platform/ui/color';
import { DEFAULT_COLOR_SWATCHES, recordRecentColor, useRecentColors } from '@platform/ui/colorPickerStore';
import { Scrollable } from '@platform/ui/Scrollable';
import { Tooltip } from '@platform/ui/Tooltip';
import { useCanvasEngine } from '@workbench/canvas-operations/react';
import { MAX_COLOR_PALETTE_SIZE } from '@workbench/widgets/canvas/color-system/colorPair';
import {
  useActiveColorCommands,
  useActiveColorPair,
  useActiveColorTarget,
  useColorPalette,
} from '@workbench/widgets/canvas/color-system/useActiveColors';
import { useMaskTintEditor } from '@workbench/widgets/canvas/color-system/useMaskTintEditor';
import { PlusIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const SHELF_SEPARATOR = <Separator borderColor="border.subtle" />;

/** Share default, account-recent, and project swatches; picks update the same active pair as Color. */
export const SwatchesPane = () => {
  const { t } = useTranslation();
  const engine = useCanvasEngine();
  const pair = useActiveColorPair();
  const target = useActiveColorTarget();
  const palette = useColorPalette();
  const commands = useActiveColorCommands();
  const recents = useRecentColors();
  // The shelves write the pane's active target — the armed mask tint included.
  const maskTint = useMaskTintEditor(engine);
  const activeHex = maskTint ? maskTint.color : pair[target];

  const applyHex = useCallback(
    (hex: string) => {
      if (normalizeHex(hex, '') === '') {
        return;
      }
      // Flattened like the pair itself, so a pick (even of a legacy alpha
      // recent) records what it actually sets.
      const opaque = formatHexColor(parseHexColor(hex));
      if (maskTint) {
        maskTint.commit(opaque);
      } else {
        commands.setPairColor(target, opaque);
      }
      recordRecentColor(opaque);
    },
    [commands, maskTint, target]
  );
  const addToPalette = useCallback(() => commands.addPaletteColor(activeHex), [activeHex, commands]);
  const removeFromPalette = useCallback((color: string) => commands.removePaletteColor(color), [commands]);
  const canAddToPalette = !palette.includes(activeHex) && palette.length < MAX_COLOR_PALETTE_SIZE;
  const addButton = useMemo(
    () => (
      <Tooltip content={t('widgets.layers.colorPane.addToPalette')}>
        <IconButton
          aria-label={t('widgets.layers.colorPane.addToPalette')}
          color="fg.muted"
          disabled={!canAddToPalette}
          size="sm"
          variant="ghost"
          onClick={addToPalette}
        >
          <PlusIcon size={14} />
        </IconButton>
      </Tooltip>
    ),
    [addToPalette, canAddToPalette, t]
  );

  return (
    <Scrollable h="full">
      <Stack gap="2.5" p="2" separator={SHELF_SEPARATOR}>
        <SwatchShelf
          activeHex={activeHex}
          colors={DEFAULT_COLOR_SWATCHES}
          label={t('widgets.layers.colorPane.swatches')}
          onPick={applyHex}
        />
        {recents.length > 0 ? (
          <SwatchShelf
            activeHex={activeHex}
            colors={recents}
            label={t('widgets.layers.colorPane.recents')}
            onPick={applyHex}
          />
        ) : null}
        <SwatchShelf
          activeHex={activeHex}
          colors={palette}
          label={t('widgets.layers.colorPane.palette')}
          trailing={addButton}
          onPick={applyHex}
          onRemove={removeFromPalette}
        />
      </Stack>
    </Scrollable>
  );
};

const SwatchShelf = ({
  activeHex,
  colors,
  label,
  onPick,
  onRemove,
  trailing,
}: {
  activeHex: string;
  colors: readonly string[];
  label: string;
  onPick: (hex: string) => void;
  onRemove?: (hex: string) => void;
  trailing?: ReactNode;
}) => (
  <Stack gap="1.5">
    <HStack justify="space-between" minH="5">
      <Text color="fg.muted" fontSize="xs" fontWeight="600" textTransform="uppercase">
        {label}
      </Text>
      {trailing}
    </HStack>
    <Flex aria-label={label} gap="1" role="group" wrap="wrap">
      {colors.map((color) => (
        <Swatch key={color} color={color} isActive={color === activeHex} onPick={onPick} onRemove={onRemove} />
      ))}
    </Flex>
  </Stack>
);

const Swatch = ({
  color,
  isActive,
  onPick,
  onRemove,
}: {
  color: string;
  isActive: boolean;
  onPick: (hex: string) => void;
  onRemove?: (hex: string) => void;
}) => {
  const { t } = useTranslation();
  const style = useMemo(() => ({ backgroundColor: color }), [color]);
  const pick = useCallback(() => onPick(color), [color, onPick]);
  const onContextMenu = useCallback(
    (event: ReactMouseEvent<HTMLButtonElement>) => {
      if (onRemove) {
        event.preventDefault();
        onRemove(color);
      }
    },
    [color, onRemove]
  );
  const onKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLButtonElement>) => {
      if (onRemove && (event.key === 'Delete' || event.key === 'Backspace')) {
        event.preventDefault();
        // The button unmounts on removal; keep the keyboard user on the shelf.
        const sibling = (event.currentTarget.nextElementSibling ??
          event.currentTarget.previousElementSibling) as HTMLElement | null;
        sibling?.focus();
        onRemove(color);
      }
    },
    [color, onRemove]
  );

  return (
    <chakra.button
      aria-label={color}
      aria-pressed={isActive}
      borderColor={isActive ? 'accent.solid' : 'border.subtle'}
      borderWidth="1px"
      h="6"
      rounded="sm"
      style={style}
      title={onRemove ? t('widgets.layers.colorPane.removeSwatchHint', { color }) : color}
      type="button"
      w="6"
      onClick={pick}
      onContextMenu={onContextMenu}
      onKeyDown={onKeyDown}
    />
  );
};
