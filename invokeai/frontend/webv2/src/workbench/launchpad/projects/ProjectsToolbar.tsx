import { Flex, Icon, Input, InputGroup, Menu, Portal } from '@chakra-ui/react';
import { Button, IconButton } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { Tooltip } from '@platform/ui/Tooltip';
import { ArrowUpDownIcon, CheckIcon, LayoutGridIcon, ListIcon, SearchIcon, XIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

/* eslint-disable react-perf/jsx-no-jsx-as-prop */
import type { ProjectSortId, ProjectsViewId } from './projectLibraryView';

import { PROJECT_SORT_IDS } from './projectLibraryView';

/** Search, ordering, and layout for the project library. */

const MENU_POSITIONING = { placement: 'bottom-end' } as const;
const SEARCH_ICON = <Icon as={SearchIcon} boxSize="3.5" color="fg.subtle" />;

const SORT_LABEL_KEY: Record<ProjectSortId, string> = {
  created: 'projects.sort.created',
  edited: 'projects.sort.edited',
  name: 'projects.sort.name',
};

export const ProjectsToolbar = ({
  searchTerm,
  sort,
  view,
  onSearchTermChange,
  onSortChange,
  onViewChange,
}: {
  searchTerm: string;
  sort: ProjectSortId;
  view: ProjectsViewId;
  onSearchTermChange: (value: string) => void;
  onSortChange: (sort: ProjectSortId) => void;
  onViewChange: (view: ProjectsViewId) => void;
}) => {
  const { t } = useTranslation();

  const handleSearchInput = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => onSearchTermChange(event.target.value),
    [onSearchTermChange]
  );
  const handleClearSearch = useCallback(() => onSearchTermChange(''), [onSearchTermChange]);
  const handleSortSelect = useCallback(
    (details: { value: string }) => {
      if (PROJECT_SORT_IDS.includes(details.value as ProjectSortId)) {
        onSortChange(details.value as ProjectSortId);
      }
    },
    [onSortChange]
  );
  const handleGridView = useCallback(() => onViewChange('grid'), [onViewChange]);
  const handleListView = useCallback(() => onViewChange('list'), [onViewChange]);

  return (
    <Flex align="center" gap="2" wrap="wrap">
      <InputGroup
        endElement={
          searchTerm ? (
            <IconButton aria-label={t('common.clearSearch')} size="2xs" variant="ghost" onClick={handleClearSearch}>
              <Icon as={XIcon} boxSize="3" />
            </IconButton>
          ) : undefined
        }
        flex="1"
        maxW="xs"
        minW="3xs"
        startElement={SEARCH_ICON}
      >
        <Input
          aria-label={t('projects.searchLabel')}
          placeholder={t('projects.searchPlaceholder')}
          size="xs"
          type="search"
          value={searchTerm}
          onChange={handleSearchInput}
        />
      </InputGroup>

      <Menu.Root positioning={MENU_POSITIONING} onSelect={handleSortSelect}>
        <Menu.Trigger asChild>
          <Button size="xs" variant="outline">
            <Icon as={ArrowUpDownIcon} boxSize="3.5" />
            {t(SORT_LABEL_KEY[sort])}
          </Button>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="44">
              <Menu.ItemGroup>
                <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                  {t('projects.sort.label')}
                </Menu.ItemGroupLabel>
                {PROJECT_SORT_IDS.map((id) => (
                  <Menu.Item key={id} value={id}>
                    <Icon as={CheckIcon} boxSize="3.5" opacity={id === sort ? 1 : 0} />
                    <Menu.ItemText>{t(SORT_LABEL_KEY[id])}</Menu.ItemText>
                  </Menu.Item>
                ))}
              </Menu.ItemGroup>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>

      <Flex gap="0.5">
        <Tooltip content={t('projects.gridView')}>
          <IconButton
            aria-label={t('projects.gridView')}
            aria-pressed={view === 'grid'}
            size="xs"
            variant={view === 'grid' ? 'subtle' : 'ghost'}
            onClick={handleGridView}
          >
            <Icon as={LayoutGridIcon} boxSize="3.5" />
          </IconButton>
        </Tooltip>
        <Tooltip content={t('projects.listView')}>
          <IconButton
            aria-label={t('projects.listView')}
            aria-pressed={view === 'list'}
            size="xs"
            variant={view === 'list' ? 'subtle' : 'ghost'}
            onClick={handleListView}
          >
            <Icon as={ListIcon} boxSize="3.5" />
          </IconButton>
        </Tooltip>
      </Flex>
    </Flex>
  );
};
