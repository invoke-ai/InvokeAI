import type { TransformAnchor } from '@workbench/canvas-engine/api';
import type { KeyboardEvent } from 'react';

import { Box, Grid, Popover, Portal } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { PopoverContent } from '@platform/ui/Popover';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

/** Row-major: the arrow keys derive row and column from the index. */
const ANCHORS: readonly TransformAnchor[] = ['nw', 'n', 'ne', 'w', 'center', 'e', 'sw', 's', 'se'];

const ANCHOR_LABEL_KEYS: Record<TransformAnchor, string> = {
  center: 'widgets.transform.anchors.center',
  e: 'widgets.transform.anchors.right',
  n: 'widgets.transform.anchors.top',
  ne: 'widgets.transform.anchors.topRight',
  nw: 'widgets.transform.anchors.topLeft',
  s: 'widgets.transform.anchors.bottom',
  se: 'widgets.transform.anchors.bottomRight',
  sw: 'widgets.transform.anchors.bottomLeft',
  w: 'widgets.transform.anchors.left',
};

const STEPS: Record<string, [row: number, column: number]> = {
  ArrowDown: [1, 0],
  ArrowLeft: [0, -1],
  ArrowRight: [0, 1],
  ArrowUp: [-1, 0],
};

const POSITIONING = { placement: 'bottom-end' } as const;

const clampCell = (value: number): number => Math.min(2, Math.max(0, value));

const ReferencePointGlyph = ({ value }: { value: TransformAnchor }) => (
  <svg aria-hidden fill="currentColor" height="16" viewBox="0 0 16 16" width="16">
    {ANCHORS.map((anchor, index) => {
      const cx = 3 + (index % 3) * 5;
      const cy = 3 + Math.floor(index / 3) * 5;
      return anchor === value ? (
        <rect key={anchor} height="5" rx="1" width="5" x={cx - 2.5} y={cy - 2.5} />
      ) : (
        <circle key={anchor} cx={cx} cy={cy} opacity="0.55" r="1.25" />
      );
    })}
  </svg>
);

const ReferencePointCell = ({
  anchor,
  checked,
  label,
  onChoose,
}: {
  anchor: TransformAnchor;
  checked: boolean;
  label: string;
  onChoose: (anchor: TransformAnchor) => void;
}) => {
  const onClick = useCallback(() => onChoose(anchor), [anchor, onChoose]);
  return (
    <IconButton
      aria-checked={checked}
      aria-label={label}
      colorPalette={checked ? 'accent' : undefined}
      data-anchor={anchor}
      role="radio"
      tabIndex={checked ? 0 : -1}
      tooltip={label}
      variant={checked ? 'solid' : 'ghost'}
      onClick={onClick}
    >
      <Box
        bg={checked ? 'currentColor' : 'fg.muted'}
        boxSize={checked ? '2.5' : '2'}
        rounded={checked ? 'xs' : 'full'}
      />
    </IconButton>
  );
};

export const ReferencePointPicker = ({
  value,
  onChange,
}: {
  value: TransformAnchor;
  onChange: (anchor: TransformAnchor) => void;
}) => {
  const { t } = useTranslation();
  const triggerIds = useTooltipTriggerIds();
  const [open, setOpen] = useState(false);
  const triggerLabel = t('widgets.transform.referencePointValue', { position: t(ANCHOR_LABEL_KEYS[value]) });
  const onOpenChange = useCallback(({ open: next }: { open: boolean }) => setOpen(next), []);
  const choose = useCallback(
    (anchor: TransformAnchor) => {
      onChange(anchor);
      setOpen(false);
    },
    [onChange]
  );
  const onKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      const step = STEPS[event.key];
      if (!step) {
        return;
      }
      event.preventDefault();
      const index = ANCHORS.indexOf(value);
      const next = clampCell(Math.floor(index / 3) + step[0]) * 3 + clampCell((index % 3) + step[1]);
      if (next !== index) {
        onChange(ANCHORS[next]!);
        event.currentTarget.querySelector<HTMLButtonElement>(`[data-anchor="${ANCHORS[next]}"]`)?.focus();
      }
    },
    [onChange, value]
  );
  return (
    <Popover.Root
      ids={triggerIds}
      lazyMount
      open={open}
      positioning={POSITIONING}
      unmountOnExit
      onOpenChange={onOpenChange}
    >
      <Tooltip content={triggerLabel} disabled={open} ids={triggerIds} keepsRootWhenDisabled>
        <Popover.Trigger asChild>
          <IconButton aria-label={triggerLabel} color="fg" flexShrink={0} variant="outline">
            <ReferencePointGlyph value={value} />
          </IconButton>
        </Popover.Trigger>
      </Tooltip>
      <Portal>
        <Popover.Positioner>
          <PopoverContent p="1.5" w="auto">
            <Grid
              aria-label={t('widgets.transform.referencePoint')}
              gap="0.5"
              role="radiogroup"
              templateColumns="repeat(3, auto)"
              onKeyDown={onKeyDown}
            >
              {ANCHORS.map((anchor) => (
                <ReferencePointCell
                  key={anchor}
                  anchor={anchor}
                  checked={anchor === value}
                  label={t(ANCHOR_LABEL_KEYS[anchor])}
                  onChoose={choose}
                />
              ))}
            </Grid>
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};
