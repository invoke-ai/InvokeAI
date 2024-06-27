import type { ComboboxOption } from '@invoke-ai/ui-library';
import {
  Button,
  ButtonGroup,
  Combobox,
  Flex,
  FormControl,
  FormLabel,
  IconButton,
  Popover,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { SingleValue } from 'chakra-react-select';
import { orderByChanged, orderDirChanged } from 'features/gallery/store/gallerySlice';
import type { OrderBy, OrderDir } from 'features/gallery/store/types';
import { t } from 'i18next';
import { useCallback, useMemo } from 'react';
import { PiSortAscending, PiSortDescending } from 'react-icons/pi';

const OPTIONS = [
  { value: 'created_at', label: t('gallery.createdDate') },
  { value: 'starred', label: t('gallery.starred') },
];

export const GallerySort = () => {
  const { orderBy, orderDir } = useAppSelector((s) => s.gallery);
  const dispatch = useAppDispatch();

  const handleChangeOrderDir = useCallback(
    (dir: OrderDir) => {
      dispatch(orderDirChanged(dir));
    },
    [dispatch]
  );

  const handleChangeOrderBy = useCallback(
    (v: SingleValue<ComboboxOption>) => {
      if (v) {
        dispatch(orderByChanged(v.value as OrderBy));
      }
    },
    [dispatch]
  );

  const orderByValue = useMemo(() => {
    return OPTIONS.find((opt) => opt.value === orderBy);
  }, [orderBy]);

  const ascendingText = useMemo(() => {
    return orderBy === 'created_at' ? t('gallery.oldestFirst') : t('gallery.starredLast');
  }, [orderBy]);

  const descendingText = useMemo(() => {
    return orderBy === 'created_at' ? t('gallery.newestFirst') : t('gallery.starredFirst');
  }, [orderBy]);

  const sortTooltip = useMemo(() => {
    if (orderDir === 'ASC') {
      return `${t('gallery.sortingBy')}: ${ascendingText}`;
    } else {
      return `${t('gallery.sortingBy')}: ${descendingText}`;
    }
  }, [orderDir, ascendingText, descendingText]);

  return (
    <Popover isLazy>
      <PopoverTrigger>
        <IconButton
          tooltip={sortTooltip}
          variant="outline"
          size="sm"
          icon={orderDir === 'ASC' ? <PiSortAscending /> : <PiSortDescending />}
          aria-label="Sort"
        />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody>
          <Flex direction="column" gap={4}>
            <ButtonGroup>
              <Button
                size="sm"
                flexShrink={0}
                onClick={handleChangeOrderDir.bind(null, 'DESC')}
                colorScheme={orderDir === 'DESC' ? 'invokeBlue' : 'base'}
              >
                {descendingText}
              </Button>
              <Button
                size="sm"
                flexShrink={0}
                onClick={handleChangeOrderDir.bind(null, 'ASC')}
                colorScheme={orderDir === 'ASC' ? 'invokeBlue' : 'base'}
              >
                {ascendingText}
              </Button>
            </ButtonGroup>
            <FormControl>
              <FormLabel>{t('gallery.sortBy')}</FormLabel>
              <Combobox value={orderByValue} options={OPTIONS} onChange={handleChangeOrderBy} isSearchable={false} />
            </FormControl>
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};
