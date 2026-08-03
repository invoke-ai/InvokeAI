import type { GalleryBoardOrderBy, GalleryOrderDir } from '@features/gallery/core/types';

import { HStack, Icon, Input, InputGroup, Menu, Portal } from '@chakra-ui/react';
import { CloseButton, IconButton, ToggleIconButton } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { Tooltip } from '@platform/ui/Tooltip';
import { ArchiveIcon, ArrowUpDownIcon, CalendarIcon, SearchIcon } from 'lucide-react';
import { useCallback, useMemo, type ChangeEvent, type KeyboardEvent, type Ref } from 'react';
import { useTranslation } from 'react-i18next';

import { useMenuTriggerIds } from './galleryMenuIds';
import { useGalleryWidget } from './GalleryWidgetContext';

const SEARCH_START_ELEMENT = <Icon as={SearchIcon} size="xs" />;
const SORT_MENU_POSITIONING = { placement: 'bottom-end' } as const;

/**
 * Board search plus the three list controls. All three are icon-only in both
 * layouts — the wide sidebar has no room for labels, and giving the stacked
 * layout labelled variants would be exactly the size-conditional behaviour this
 * redesign exists to remove.
 */
export const GalleryBoardFilters = ({
  ref,
  searchTerm,
  onSearchChange,
  onSubmitSearch,
}: {
  /** The panel focuses the field when "+" is pressed with nothing typed. */
  ref?: Ref<HTMLInputElement>;
  searchTerm: string;
  onSearchChange: (searchTerm: string) => void;
  /** Enter on the field; the panel decides whether that means "create". */
  onSubmitSearch: () => void;
}) => {
  const { t } = useTranslation();
  const { actions, gallery } = useGalleryWidget();
  const { boardOrderBy, boardOrderDir, showArchivedBoards, showDateBoards } = gallery.settings;
  const sortTriggerIds = useMenuTriggerIds();

  const handleSearchChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => onSearchChange(event.currentTarget.value),
    [onSearchChange]
  );

  const handleClearSearch = useCallback(() => onSearchChange(''), [onSearchChange]);

  const handleSearchKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      event.stopPropagation();

      if (event.key === 'Enter') {
        event.preventDefault();
        onSubmitSearch();
      }
    },
    [onSubmitSearch]
  );

  const handleToggleDateBoards = useCallback(
    (showDateBoards: boolean) => actions.updateSettings({ showDateBoards }),
    [actions]
  );

  const handleToggleArchivedBoards = useCallback(
    (showArchivedBoards: boolean) => actions.updateSettings({ showArchivedBoards }),
    [actions]
  );

  const handleOrderByChange = useCallback(
    (event: { value: string }) => actions.updateSettings({ boardOrderBy: event.value as GalleryBoardOrderBy }),
    [actions]
  );

  const handleOrderDirChange = useCallback(
    (event: { value: string }) => actions.updateSettings({ boardOrderDir: event.value as GalleryOrderDir }),
    [actions]
  );

  const clearSearchButton = useMemo(
    () =>
      searchTerm ? (
        <CloseButton aria-label={t('common.clearSearch')} me="-2" size="2xs" onClick={handleClearSearch} />
      ) : null,
    [searchTerm, handleClearSearch, t]
  );

  return (
    <HStack gap="1">
      <InputGroup endElement={clearSearchButton} flex="1" minW="0" startElement={SEARCH_START_ELEMENT}>
        <Input
          ref={ref}
          aria-label={t('widgets.gallery.searchOrCreateBoards')}
          placeholder={t('widgets.gallery.searchOrCreateBoards')}
          size="xs"
          value={searchTerm}
          onChange={handleSearchChange}
          onKeyDown={handleSearchKeyDown}
        />
      </InputGroup>
      <ToggleIconButton
        checked={showDateBoards}
        icon={CalendarIcon}
        size="xs"
        label={showDateBoards ? t('widgets.gallery.hideDateBoards') : t('widgets.gallery.showDateBoards')}
        onCheckedChange={handleToggleDateBoards}
      />
      <ToggleIconButton
        checked={showArchivedBoards}
        icon={ArchiveIcon}
        size="xs"
        label={showArchivedBoards ? t('widgets.gallery.hideArchivedBoards') : t('widgets.gallery.showArchivedBoards')}
        onCheckedChange={handleToggleArchivedBoards}
      />
      <Menu.Root ids={sortTriggerIds} positioning={SORT_MENU_POSITIONING}>
        <Tooltip content={t('widgets.gallery.sortBoards')} ids={sortTriggerIds}>
          <Menu.Trigger asChild>
            <IconButton aria-label={t('widgets.gallery.sortBoards')} color="fg.muted" size="xs" variant="ghost">
              <Icon as={ArrowUpDownIcon} boxSize="3.5" />
            </IconButton>
          </Menu.Trigger>
        </Tooltip>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="9rem">
              <Menu.ItemGroup>
                <Menu.ItemGroupLabel>{t('widgets.gallery.sortBoardsBy')}</Menu.ItemGroupLabel>
                <Menu.RadioItemGroup value={boardOrderBy} onValueChange={handleOrderByChange}>
                  <Menu.RadioItem value="created_at">
                    <Menu.ItemIndicator />
                    <Menu.ItemText>{t('common.created')}</Menu.ItemText>
                  </Menu.RadioItem>
                  <Menu.RadioItem value="board_name">
                    <Menu.ItemIndicator />
                    <Menu.ItemText>{t('common.name')}</Menu.ItemText>
                  </Menu.RadioItem>
                </Menu.RadioItemGroup>
              </Menu.ItemGroup>
              <Menu.Separator />
              <Menu.ItemGroup>
                <Menu.RadioItemGroup value={boardOrderDir} onValueChange={handleOrderDirChange}>
                  <Menu.RadioItem value="DESC">
                    <Menu.ItemIndicator />
                    <Menu.ItemText>{t('widgets.gallery.sortBoardsDescending')}</Menu.ItemText>
                  </Menu.RadioItem>
                  <Menu.RadioItem value="ASC">
                    <Menu.ItemIndicator />
                    <Menu.ItemText>{t('widgets.gallery.sortBoardsAscending')}</Menu.ItemText>
                  </Menu.RadioItem>
                </Menu.RadioItemGroup>
              </Menu.ItemGroup>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
    </HStack>
  );
};
