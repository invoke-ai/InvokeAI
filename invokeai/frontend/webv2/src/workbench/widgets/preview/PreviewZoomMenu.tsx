import { HStack, Menu, Portal, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { MenuActionItem, MenuContent } from '@platform/ui/Menu';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { ChevronDownIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { PreviewZoomControls } from './previewHeaderStore';

const ZOOM_PRESETS = [100, 200, 400] as const;
const MENU_POSITIONING = { placement: 'bottom-start' } as const;

/**
 * Expose fitted/actual zoom and keyboard presets in the header; the stage owns wheel, pinch, and double-click
 * gestures.
 */
export const PreviewZoomMenu = ({ zoom }: { zoom: PreviewZoomControls }) => {
  const { t } = useTranslation();
  const percent = zoom.percent;
  const label = percent === null ? '—' : t('widgets.preview.zoomPercent', { percent });
  const ids = useTooltipTriggerIds();

  return (
    <Menu.Root ids={ids} positioning={MENU_POSITIONING}>
      <Tooltip content={t('widgets.preview.zoom')} ids={ids}>
        <Menu.Trigger asChild>
          <Button
            aria-label={percent === null ? t('widgets.preview.zoom') : t('widgets.preview.zoomLevel', { percent })}
            color="fg.muted"
            fontVariantNumeric="tabular-nums"
            minW="0"
            px="1.5"
            size="2xs"
            variant="ghost"
          >
            <HStack gap="1">
              <Text fontSize="xs" fontWeight="600">
                {label}
              </Text>
              <ChevronDownIcon size={12} />
            </HStack>
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
                // A preset the fit already exceeds would only snap back to fit.
                disabled={zoom.fitPercent !== null && preset <= zoom.fitPercent}
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
