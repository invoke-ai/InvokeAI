import type { ActiveColorTarget } from '@workbench/widgets/canvas/color-system/colorPair';
import type { KeyboardEvent as ReactKeyboardEvent } from 'react';

import { Box, chakra, Flex, HStack, Input, Stack } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { formatHexColor, normalizeHex, parseHexColor, type RgbaColor } from '@platform/ui/color';
import {
  COLOR_PICKER_FORMATS,
  recordRecentColor,
  setColorPickerFormat,
  setColorPickerMode,
  useColorPickerFormat,
  useColorPickerMode,
  type ColorPickerFormat,
} from '@platform/ui/colorPickerStore';
import { hexToHsv, hslToRgb, hsvToHex, rgbToHsl, type HsvColor } from '@platform/ui/hsv';
import { HsvBoxPicker } from '@platform/ui/HsvBoxPicker';
import { HsvWheelPicker } from '@platform/ui/HsvWheelPicker';
import { Scrollable } from '@platform/ui/Scrollable';
import { Tooltip } from '@platform/ui/Tooltip';
import { getDocumentLayer } from '@workbench/canvas-engine/api';
import { useCanvasEngine } from '@workbench/canvas-operations/react';
import { armMaskTintTarget, clearMaskTintTarget } from '@workbench/widgets/canvas/color-system/maskTintTarget';
import {
  useActiveColorCommands,
  useActiveColorPair,
  useActiveColorTarget,
} from '@workbench/widgets/canvas/color-system/useActiveColors';
import { useMaskTintEditor, type MaskTintEditor } from '@workbench/widgets/canvas/color-system/useMaskTintEditor';
import { FormNumberField } from '@workbench/widgets/canvas/tool-presentation/FormControls';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { ArrowLeftRightIcon, CircleIcon, PipetteIcon, RotateCcwIcon, SquareIcon } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

const FORMAT_HOVER_PROPS = { bg: 'bg.muted', color: 'fg' };

/**
 * Edit one persisted foreground/background pair using the user's preferred wheel/box and shared channels. Keep
 * only sticky HSV locally to preserve hue through greys; shelves live in Swatches.
 */
export const ColorPane = () => {
  const { t } = useTranslation();
  const engine = useCanvasEngine();
  const pair = useActiveColorPair();
  const target = useActiveColorTarget();
  const commands = useActiveColorCommands();
  const mode = useColorPickerMode();
  // The explicit non-pair target: the selected mask's tint, when armed. Its
  // edits commit through the document seam — live preview, one entry a gesture.
  const maskTint = useMaskTintEditor(engine);
  const selectedMask = useActiveProjectSelector((project) => {
    const { document } = project.canvas;
    const layer = document.selectedLayerId ? getDocumentLayer(document, document.selectedLayerId) : null;
    return layer && (layer.type === 'inpaint_mask' || layer.type === 'regional_guidance')
      ? { color: layer.mask.fill.color, id: layer.id }
      : null;
  }, areSelectedMasksEqual);
  const activeHex = maskTint ? maskTint.color : pair[target];

  // The picker holds HSV so hue survives greys; it resyncs from the pair only
  // when the pair changed to something the current HSV does not already mean.
  const [hsv, setHsv] = useState<HsvColor>(() => hexToHsv(activeHex));
  const [syncedHex, setSyncedHex] = useState(activeHex);
  if (activeHex !== syncedHex) {
    setSyncedHex(activeHex);
    if (hsvToHex(hsv) !== activeHex) {
      setHsv(hexToHsv(activeHex, hsv));
    }
  }

  const applyHsv = useCallback(
    (next: HsvColor) => {
      setHsv(next);
      const hex = hsvToHex(next);
      setSyncedHex(hex);
      if (maskTint) {
        maskTint.preview(hex);
        return;
      }
      commands.setPairColor(target, hex);
    },
    [commands, maskTint, target]
  );
  // Record only settled arrow-repeat colors; immediate commits supersede pending recents and unmount flushes them.
  const pendingRecent = useRef<{ hex: string; timer: ReturnType<typeof setTimeout> } | null>(null);
  const settlePendingRecent = useCallback((record: boolean) => {
    if (!pendingRecent.current) {
      return;
    }
    clearTimeout(pendingRecent.current.timer);
    if (record) {
      recordRecentColor(pendingRecent.current.hex);
    }
    pendingRecent.current = null;
  }, []);
  useEffect(() => () => settlePendingRecent(true), [settlePendingRecent]);
  const commitHsv = useCallback(
    (next: HsvColor) => {
      applyHsv(next);
      const hex = hsvToHex(next);
      maskTint?.commit(hex);
      settlePendingRecent(false);
      pendingRecent.current = { hex, timer: setTimeout(() => settlePendingRecent(true), 500) };
    },
    [applyHsv, maskTint, settlePendingRecent]
  );
  const applyHex = useCallback(
    (hex: string) => {
      if (normalizeHex(hex, '') === '') {
        return;
      }
      // Flattened like the pair itself, so a recent swatch always matches what it sets.
      const opaque = formatHexColor(parseHexColor(hex));
      settlePendingRecent(false);
      if (maskTint) {
        maskTint.commit(opaque);
      } else {
        commands.setPairColor(target, opaque);
      }
      recordRecentColor(opaque);
    },
    [commands, maskTint, settlePendingRecent, target]
  );
  const sampleFromCanvas = useCallback(async () => {
    if (!engine) {
      return;
    }
    const hex = await engine.tools.requestColorSample();
    if (hex) {
      // Routed through the command so the target is read when the sample lands,
      // not when the pipette was armed. The pane's pipette is the mask tint's
      // editor-specific sampler; the standing router never writes the tint.
      settlePendingRecent(false);
      if (maskTint) {
        maskTint.commit(hex);
      } else {
        commands.applySampledColor(hex);
      }
      recordRecentColor(hex);
    }
  }, [commands, engine, maskTint, settlePendingRecent]);
  const toggleMode = useCallback(() => setColorPickerMode(mode === 'wheel' ? 'box' : 'wheel'), [mode]);
  const modeLabel =
    mode === 'wheel' ? t('widgets.layers.colorPane.pickerModeBox') : t('widgets.layers.colorPane.pickerModeWheel');

  return (
    <Scrollable h="full">
      <Flex align="flex-start" gap="2.5" p="2">
        <Box flexShrink={0} w="168px">
          {mode === 'wheel' ? (
            <HsvWheelPicker diameterPx={168} value={hsv} onChange={applyHsv} onChangeEnd={commitHsv} />
          ) : (
            <HsvBoxPicker heightPx={140} value={hsv} onChange={applyHsv} onChangeEnd={commitHsv} />
          )}
        </Box>
        <Stack flex="1" gap="2" minW="0">
          <HStack gap="1">
            <TargetChips
              maskTint={maskTint}
              pair={pair}
              selectedMask={selectedMask}
              target={target}
              onSelectTarget={commands.setTarget}
            />
            <Tooltip content={t('widgets.canvas.commands.swapColors')}>
              <IconButton
                aria-label={t('widgets.canvas.commands.swapColors')}
                color="fg.muted"
                size="sm"
                variant="ghost"
                onClick={commands.swapPair}
              >
                <ArrowLeftRightIcon size={14} />
              </IconButton>
            </Tooltip>
            <Tooltip content={t('widgets.canvas.commands.resetColors')}>
              <IconButton
                aria-label={t('widgets.canvas.commands.resetColors')}
                color="fg.muted"
                size="sm"
                variant="ghost"
                onClick={commands.resetPair}
              >
                <RotateCcwIcon size={14} />
              </IconButton>
            </Tooltip>
            <Box flex="1" />
            {engine ? (
              <Tooltip content={t('common.colorPicker.sampleFromCanvas')}>
                <IconButton
                  aria-label={t('common.colorPicker.sampleFromCanvas')}
                  color="fg.muted"
                  size="sm"
                  variant="ghost"
                  onClick={sampleFromCanvas}
                >
                  <PipetteIcon size={14} />
                </IconButton>
              </Tooltip>
            ) : null}
            <Tooltip content={modeLabel}>
              <IconButton aria-label={modeLabel} color="fg.muted" size="sm" variant="ghost" onClick={toggleMode}>
                {mode === 'wheel' ? <SquareIcon size={14} /> : <CircleIcon size={14} />}
              </IconButton>
            </Tooltip>
          </HStack>
          <ChannelFields activeHex={activeHex} hsv={hsv} onCommitHex={applyHex} onCommitHsv={commitHsv} />
        </Stack>
      </Flex>
    </Scrollable>
  );
};

const areSelectedMasksEqual = (
  a: { color: string; id: string } | null,
  b: { color: string; id: string } | null
): boolean => a?.id === b?.id && a?.color === b?.color;

const TargetChips = ({
  maskTint,
  onSelectTarget,
  pair,
  selectedMask,
  target,
}: {
  maskTint: MaskTintEditor | null;
  onSelectTarget: (target: ActiveColorTarget) => void;
  pair: { foreground: string; background: string };
  selectedMask: { color: string; id: string } | null;
  target: ActiveColorTarget;
}) => {
  const { t } = useTranslation();
  const selectPairTarget = useCallback(
    (next: ActiveColorTarget) => {
      // A pair chip always returns the pane to the pair, disarming the tint.
      clearMaskTintTarget();
      onSelectTarget(next);
    },
    [onSelectTarget]
  );
  const armTint = useCallback(() => {
    if (selectedMask) {
      armMaskTintTarget(selectedMask.id);
    }
  }, [selectedMask]);
  return (
    <HStack flexShrink={0} gap="1" mr="1">
      <TargetChip
        colorHex={pair.foreground}
        isActive={!maskTint && target === 'foreground'}
        label={t('widgets.layers.colorPane.foreground')}
        target="foreground"
        onSelect={selectPairTarget}
      />
      <TargetChip
        colorHex={pair.background}
        isActive={!maskTint && target === 'background'}
        label={t('widgets.layers.colorPane.background')}
        target="background"
        onSelect={selectPairTarget}
      />
      {selectedMask ? (
        // Revealed by a compatible selection, activated only explicitly.
        <TargetChip
          colorHex={maskTint?.color ?? selectedMask.color}
          isActive={maskTint !== null}
          label={t('widgets.layers.colorPane.maskTint')}
          target="foreground"
          onSelect={armTint}
        />
      ) : null}
    </HStack>
  );
};

const TargetChip = ({
  colorHex,
  isActive,
  label,
  onSelect,
  target,
}: {
  colorHex: string;
  isActive: boolean;
  label: string;
  onSelect: (target: ActiveColorTarget) => void;
  target: ActiveColorTarget;
}) => {
  const style = useMemo(() => ({ backgroundColor: colorHex }), [colorHex]);
  const select = useCallback(() => onSelect(target), [onSelect, target]);
  return (
    <chakra.button
      aria-label={label}
      aria-pressed={isActive}
      borderColor={isActive ? 'accent.solid' : 'border'}
      borderWidth="2px"
      h="6"
      rounded="sm"
      style={style}
      title={colorHex}
      type="button"
      w="6"
      onClick={select}
    />
  );
};

const CHANNEL_KEYS = {
  hsb: ['hue', 'saturation', 'brightness'],
  hsl: ['hue', 'saturation', 'lightness'],
  rgb: ['red', 'green', 'blue'],
} as const;

const ChannelFields = ({
  activeHex,
  hsv,
  onCommitHex,
  onCommitHsv,
}: {
  activeHex: string;
  hsv: HsvColor;
  onCommitHex: (hex: string) => void;
  onCommitHsv: (next: HsvColor) => void;
}) => {
  const { t } = useTranslation();
  const format = useColorPickerFormat();
  const cycleFormat = useCallback(() => {
    const index = COLOR_PICKER_FORMATS.indexOf(format);
    setColorPickerFormat(COLOR_PICKER_FORMATS[(index + 1) % COLOR_PICKER_FORMATS.length]!);
  }, [format]);
  const rgb = useMemo(() => parseHexColor(activeHex), [activeHex]);
  const hsl = useMemo(() => rgbToHsl(rgb), [rgb]);
  const channelValues: Record<Exclude<ColorPickerFormat, 'hex'>, [number, number, number]> = {
    hsb: [Math.round(hsv.h) % 360, Math.round(hsv.s * 100), Math.round(hsv.v * 100)],
    hsl: [Math.round(hsl.h) % 360, Math.round(hsl.s * 100), Math.round(hsl.l * 100)],
    rgb: [rgb.r, rgb.g, rgb.b],
  };
  const commitChannel = useCallback(
    (index: number, raw: number) => {
      if (format === 'rgb') {
        const next: RgbaColor = { ...parseHexColor(activeHex) };
        const channel = (['r', 'g', 'b'] as const)[index]!;
        next[channel] = clamp(Math.round(raw), 0, 255);
        onCommitHex(formatHexColor(next));
        return;
      }
      if (format === 'hsl') {
        const current = rgbToHsl(parseHexColor(activeHex));
        const next = {
          // A grey parses hue-agnostic; seed from the sticky hue the wheel shows.
          h: index === 0 ? clamp(raw, 0, 359.999) : current.s === 0 ? hsv.h : current.h,
          l: index === 2 ? clamp(raw, 0, 100) / 100 : current.l,
          s: index === 1 ? clamp(raw, 0, 100) / 100 : current.s,
        };
        onCommitHex(formatHexColor(hslToRgb(next)));
        return;
      }
      onCommitHsv({
        h: index === 0 ? clamp(raw, 0, 359.999) : hsv.h,
        s: index === 1 ? clamp(raw, 0, 100) / 100 : hsv.s,
        v: index === 2 ? clamp(raw, 0, 100) / 100 : hsv.v,
      });
    },
    [activeHex, format, hsv, onCommitHex, onCommitHsv]
  );

  return (
    <Flex gap="1" wrap="wrap">
      <chakra.button
        aria-label={t('widgets.layers.colorPane.format')}
        color="fg.muted"
        flexShrink={0}
        fontSize="xs"
        fontWeight="700"
        px="1.5"
        rounded="sm"
        textTransform="uppercase"
        type="button"
        _hover={FORMAT_HOVER_PROPS}
        onClick={cycleFormat}
      >
        {format}
      </chakra.button>
      {format === 'hex' ? (
        <HexField hex={activeHex} onCommit={onCommitHex} />
      ) : (
        CHANNEL_KEYS[format].map((key, index) => (
          <ChannelNumber
            key={`${format}:${key}`}
            index={index}
            label={t(`widgets.layers.colorPane.channels.${key}`)}
            value={channelValues[format][index]!}
            onCommit={commitChannel}
          />
        ))
      )}
    </Flex>
  );
};

const ChannelNumber = ({
  index,
  label,
  onCommit,
  value,
}: {
  index: number;
  label: string;
  onCommit: (index: number, raw: number) => void;
  value: number;
}) => {
  const commit = useCallback(
    ({ valueAsNumber }: { value: string; valueAsNumber: number }) => {
      if (Number.isFinite(valueAsNumber)) {
        onCommit(index, valueAsNumber);
      }
    },
    [index, onCommit]
  );
  return <FormNumberField aria-label={label} label={label.slice(0, 1)} value={String(value)} onValueCommit={commit} />;
};

const HexField = ({ hex, onCommit }: { hex: string; onCommit: (hex: string) => void }) => {
  const { t } = useTranslation();
  const [draft, setDraft] = useState<string | null>(null);
  const onChange = useCallback((event: { target: { value: string } }) => setDraft(event.target.value), []);
  const commit = useCallback(() => {
    if (draft !== null && normalizeHex(draft, '') !== '') {
      onCommit(draft);
    }
    setDraft(null);
  }, [draft, onCommit]);
  const onKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLInputElement>) => {
      if (event.key === 'Enter') {
        commit();
      } else if (event.key === 'Escape') {
        setDraft(null);
      }
    },
    [commit]
  );
  return (
    <Input
      aria-label={t('widgets.layers.colorPane.channels.hex')}
      flex="1"
      fontFamily="mono"
      minW="24"
      value={draft ?? hex}
      onBlur={commit}
      onChange={onChange}
      onKeyDown={onKeyDown}
    />
  );
};
