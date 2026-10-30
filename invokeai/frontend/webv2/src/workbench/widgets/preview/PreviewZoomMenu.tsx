import { Menu, Portal, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { MenuActionItem, MenuContent } from '@platform/ui/Menu';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { ChevronDownIcon, ZoomInIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { getActualZoomLimits, type PreviewZoomControls } from './previewHeaderStore';

const ZOOM_PRESETS = [100, 200, 400] as const;
const MENU_POSITIONING = { placement: 'bottom-start' } as const;

/**
 * Expose fitted/actual zoom and keyboard presets in the header; the stage owns wheel, pinch, and double-click
 * gestures. A `full` header shows the magnifier and, only while zoomed, the percent, so the fit readout does not cost
 * the widget title its width; the tooltip and accessible name always carry the percent. A `compact` header (a narrow
 * view, down to 280px) keeps only the readout and chevron.
 */
export const PreviewZoomMenu = ({
  density = 'full',
  zoom,
}: {
  density?: 'compact' | 'full';
  zoom: PreviewZoomControls;
}) => {
  const { t } = useTranslation();
  const percent = zoom.percent;
  const label = percent === null ? '—' : t('widgets.preview.zoomPercent', { percent });
  const ids = useTooltipTriggerIds();
  const accessibleLabel = percent === null ? t('widgets.preview.zoom') : t('widgets.preview.zoomLevel', { percent });
  const showsReadout = density === 'compact' || zoom.isZoomed;

  return (
    <Menu.Root ids={ids} positioning={MENU_POSITIONING}>
      <Tooltip content={accessibleLabel} ids={ids}>
        <Menu.Trigger asChild>
          <Button
            aria-label={accessibleLabel}
            color="fg.muted"
            fontVariantNumeric="tabular-nums"
            size="sm"
            variant="ghost"
          >
            {density === 'full' ? <ZoomInIcon aria-hidden /> : null}
            {/* Wide enough for "1600%" in tabular digits, so the trigger keeps its width as the zoom changes. */}
            {showsReadout ? (
              <Text fontSize="md" minW={density === 'full' ? '6ch' : undefined} textAlign="center">
                {label}
              </Text>
            ) : null}
            <ChevronDownIcon size={12} />
          </Button>
        </Menu.Trigger>
      </Tooltip>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="9rem">
            <MenuActionItem
              disabled={!zoom.isZoomed}
              label={t('widgets.preview.zoomFit')}
              value="fit"
              onSelect={zoom.reset}
            />
            {ZOOM_PRESETS.map((preset) => (
              <ZoomPresetItem
                key={preset}
                disabled={isPresetUnreachable(preset, zoom.fitPercent)}
                isCurrent={zoom.isZoomed && percent === preset}
                percent={preset}
                onZoomTo={zoom.zoomTo}
              />
            ))}
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

/**
 * A preset outside the loupe's zoom limits cannot be reached, and the preset equal to fit is fit itself, its own menu
 * item. Below an enlarging fit, presets down to 100% stay offered. Judged on the unrounded fit: a 100.2% fit reads as
 * 100% yet is not the image's own size.
 */
const isPresetUnreachable = (preset: number, fitPercent: number | null): boolean => {
  if (fitPercent === null) {
    return false;
  }

  const { max, min } = getActualZoomLimits(fitPercent / 100);
  const actualZoom = preset / 100;

  return preset === fitPercent || actualZoom < min || actualZoom > max;
};

const ZoomPresetItem = ({
  disabled,
  isCurrent,
  onZoomTo,
  percent,
}: {
  disabled: boolean;
  isCurrent: boolean;
  onZoomTo: (actualZoom: number) => void;
  percent: number;
}) => {
  const { t } = useTranslation();
  const handleSelect = useCallback(() => onZoomTo(percent / 100), [onZoomTo, percent]);

  return (
    <MenuActionItem
      disabled={disabled || isCurrent}
      label={t('widgets.preview.zoomPercent', { percent })}
      value={`zoom-${percent}`}
      onSelect={handleSelect}
    />
  );
};
