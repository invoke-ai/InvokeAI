import type { GallerySettings } from '@features/gallery/core/settings';

import { HStack, Icon, Menu, Portal, Slider, Stack, Switch, Text } from '@chakra-ui/react';
import { Button, IconButton } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { Tooltip } from '@platform/ui/Tooltip';
import { SettingsIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useMenuTriggerIds } from './galleryMenuIds';

const SETTINGS_POSITIONING = { placement: 'bottom-end' } as const;

/**
 * Presentation settings for the grid. Sort moved out to its own header control
 * — it is a per-look decision, not something to go hunting in a popover for.
 *
 * Driven by props rather than the widget context: the widget frame renders
 * header chrome outside the view, so in panel regions this menu mounts where
 * `useGalleryWidget` does not reach.
 */
export const GallerySettingsMenu = ({
  settings,
  onUpdateSettings,
}: {
  settings: GallerySettings;
  onUpdateSettings: (settings: Partial<GallerySettings>) => void;
}) => {
  const { t } = useTranslation();
  const { imageDensityPercent, paginationMode, showImageDimensions, showPendingItems, thumbnailFit } = settings;
  const imageSizePercent = 100 - imageDensityPercent;

  const handleSizeKeyDown = useCallback((event: React.KeyboardEvent) => event.stopPropagation(), []);
  const handleSizeChange = useCallback(
    (event: { value: number[] }) =>
      onUpdateSettings({ imageDensityPercent: 100 - (event.value[0] ?? imageSizePercent) }),
    [imageSizePercent, onUpdateSettings]
  );
  const handleSquareFit = useCallback(() => onUpdateSettings({ thumbnailFit: 'square' }), [onUpdateSettings]);
  const handleAspectFit = useCallback(() => onUpdateSettings({ thumbnailFit: 'aspect' }), [onUpdateSettings]);
  const handleShowDimensionsChange = useCallback(
    (showImageDimensions: boolean) => onUpdateSettings({ showImageDimensions }),
    [onUpdateSettings]
  );
  const handleShowPendingItemsChange = useCallback(
    (showPendingItems: boolean) => onUpdateSettings({ showPendingItems }),
    [onUpdateSettings]
  );
  const handleInfinitePagination = useCallback(
    () => onUpdateSettings({ paginationMode: 'infinite' }),
    [onUpdateSettings]
  );
  const handlePaginatedPagination = useCallback(
    () => onUpdateSettings({ paginationMode: 'paginated' }),
    [onUpdateSettings]
  );
  const sizeAriaLabel = useMemo(() => [t('widgets.gallery.imageSize')], [t]);
  const sizeValue = useMemo(() => [imageSizePercent], [imageSizePercent]);
  const triggerIds = useMenuTriggerIds();

  return (
    <Menu.Root ids={triggerIds} positioning={SETTINGS_POSITIONING}>
      <Tooltip content={t('widgets.gallery.settings')} ids={triggerIds}>
        <Menu.Trigger asChild>
          <IconButton aria-label={t('widgets.gallery.settings')} color="fg.muted" size="2xs" variant="ghost">
            <Icon as={SettingsIcon} boxSize="3.5" />
          </IconButton>
        </Menu.Trigger>
      </Tooltip>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="14rem" p="3">
            <Stack gap="3">
              <Stack gap="2">
                <HStack justify="space-between">
                  <SettingLabel>{t('widgets.gallery.settingsGridSize')}</SettingLabel>
                  <Text color="fg.subtle" fontSize="2xs">
                    {imageSizePercent}%
                  </Text>
                </HStack>
                <Slider.Root
                  aria-label={sizeAriaLabel}
                  max={100}
                  min={0}
                  step={1}
                  value={sizeValue}
                  onKeyDown={handleSizeKeyDown}
                  onValueChange={handleSizeChange}
                >
                  <Slider.Control>
                    <Slider.Track>
                      <Slider.Range />
                    </Slider.Track>
                    <Slider.Thumbs />
                  </Slider.Control>
                </Slider.Root>
              </Stack>
              <Stack gap="2">
                <SettingLabel>{t('widgets.gallery.settingsThumbnails')}</SettingLabel>
                <HStack gap="1">
                  <Button
                    flex="1"
                    size="2xs"
                    variant={thumbnailFit === 'square' ? 'solid' : 'outline'}
                    onClick={handleSquareFit}
                  >
                    {t('widgets.gallery.thumbnailFitSquare')}
                  </Button>
                  <Button
                    flex="1"
                    size="2xs"
                    variant={thumbnailFit === 'aspect' ? 'solid' : 'outline'}
                    onClick={handleAspectFit}
                  >
                    {t('widgets.gallery.thumbnailFitAspect')}
                  </Button>
                </HStack>
                <SettingSwitch
                  checked={showImageDimensions}
                  label={t('widgets.gallery.alwaysShowDimensions')}
                  onChange={handleShowDimensionsChange}
                />
                <SettingSwitch
                  checked={showPendingItems}
                  label={t('widgets.gallery.showPendingItems')}
                  onChange={handleShowPendingItemsChange}
                />
              </Stack>
              <Stack gap="2">
                <SettingLabel>{t('widgets.gallery.pagination')}</SettingLabel>
                <HStack gap="1">
                  <Button
                    flex="1"
                    size="2xs"
                    variant={paginationMode === 'infinite' ? 'solid' : 'outline'}
                    onClick={handleInfinitePagination}
                  >
                    {t('widgets.gallery.infinite')}
                  </Button>
                  <Button
                    flex="1"
                    size="2xs"
                    variant={paginationMode === 'paginated' ? 'solid' : 'outline'}
                    onClick={handlePaginatedPagination}
                  >
                    {t('common.pages')}
                  </Button>
                </HStack>
              </Stack>
            </Stack>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

const SettingSwitch = ({
  checked,
  label,
  onChange,
}: {
  checked: boolean;
  label: string;
  onChange: (checked: boolean) => void;
}) => {
  const handleCheckedChange = useCallback((event: { checked: boolean }) => onChange(event.checked), [onChange]);

  return (
    <Switch.Root checked={checked} size="sm" onCheckedChange={handleCheckedChange}>
      <Switch.HiddenInput />
      <Switch.Control>
        <Switch.Thumb />
      </Switch.Control>
      <Switch.Label fontSize="2xs">{label}</Switch.Label>
    </Switch.Root>
  );
};

const SettingLabel = ({ children }: { children: string }) => (
  <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
    {children}
  </Text>
);
