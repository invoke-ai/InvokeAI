import type { ComboboxOption, FormLabelProps } from '@invoke-ai/ui-library';
import {
  Checkbox,
  Combobox,
  CompositeSlider,
  Divider,
  Flex,
  FormControl,
  FormControlGroup,
  FormLabel,
  IconButton,
  Popover,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Switch,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { SingleValue } from 'chakra-react-select';
import {
  alwaysShowImageSizeBadgeChanged,
  autoAssignBoardOnClickChanged,
  orderDirChanged,
  setGalleryImageMinimumWidth,
  shouldAutoSwitchChanged,
  shouldShowArchivedBoardsChanged,
  starredFirstChanged,
} from 'features/gallery/store/gallerySlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { RiSettings4Fill } from 'react-icons/ri';
import { assert } from 'tsafe';

import BoardAutoAddSelect from './Boards/BoardAutoAddSelect';

const formLabelProps: FormLabelProps = {
  flexGrow: 1,
};

const GallerySettingsPopover = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const galleryImageMinimumWidth = useAppSelector((s) => s.gallery.galleryImageMinimumWidth);
  const shouldAutoSwitch = useAppSelector((s) => s.gallery.shouldAutoSwitch);
  const autoAssignBoardOnClick = useAppSelector((s) => s.gallery.autoAssignBoardOnClick);
  const alwaysShowImageSizeBadge = useAppSelector((s) => s.gallery.alwaysShowImageSizeBadge);
  const shouldShowArchivedBoards = useAppSelector((s) => s.gallery.shouldShowArchivedBoards);
  const orderDir = useAppSelector((s) => s.gallery.orderDir);
  const starredFirst = useAppSelector((s) => s.gallery.starredFirst);

  const handleChangeGalleryImageMinimumWidth = useCallback(
    (v: number) => {
      dispatch(setGalleryImageMinimumWidth(v));
    },
    [dispatch]
  );

  const handleChangeAutoSwitch = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(shouldAutoSwitchChanged(e.target.checked));
    },
    [dispatch]
  );

  const handleChangeAutoAssignBoardOnClick = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(autoAssignBoardOnClickChanged(e.target.checked)),
    [dispatch]
  );

  const handleChangeAlwaysShowImageSizeBadgeChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(alwaysShowImageSizeBadgeChanged(e.target.checked)),
    [dispatch]
  );

  const handleChangeShouldShowArchivedBoardsChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(shouldShowArchivedBoardsChanged(e.target.checked));
    },
    [dispatch]
  );

  const onChangeStarredFirst = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(starredFirstChanged(e.target.checked));
    },
    [dispatch]
  );

  const orderDirOptions = useMemo<ComboboxOption[]>(
    () => [
      { value: 'DESC', label: t('gallery.newestFirst') },
      { value: 'ASC', label: t('gallery.oldestFirst') },
    ],
    [t]
  );

  const onChangeOrderDir = useCallback(
    (v: SingleValue<ComboboxOption>) => {
      assert(v?.value === 'ASC' || v?.value === 'DESC');
      dispatch(orderDirChanged(v.value));
    },
    [dispatch]
  );

  const orderDirValue = useMemo(() => {
    return orderDirOptions.find((opt) => opt.value === orderDir);
  }, [orderDir, orderDirOptions]);

  return (
    <Popover isLazy>
      <PopoverTrigger>
        <IconButton aria-label={t('gallery.gallerySettings')} size="sm" icon={<RiSettings4Fill />} />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody>
          <Flex direction="column" gap={2}>
            <FormControl>
              <FormLabel>{t('gallery.galleryImageSize')}</FormLabel>
              <CompositeSlider
                value={galleryImageMinimumWidth}
                onChange={handleChangeGalleryImageMinimumWidth}
                min={45}
                max={256}
                defaultValue={90}
              />
            </FormControl>
            <FormControlGroup formLabelProps={formLabelProps}>
              <FormControl>
                <FormLabel>{t('gallery.autoSwitchNewImages')}</FormLabel>
                <Checkbox isChecked={shouldAutoSwitch} onChange={handleChangeAutoSwitch} />
              </FormControl>
              <FormControl>
                <FormLabel>{t('gallery.autoAssignBoardOnClick')}</FormLabel>
                <Checkbox isChecked={autoAssignBoardOnClick} onChange={handleChangeAutoAssignBoardOnClick} />
              </FormControl>
              <FormControl>
                <FormLabel>{t('gallery.alwaysShowImageSizeBadge')}</FormLabel>
                <Checkbox isChecked={alwaysShowImageSizeBadge} onChange={handleChangeAlwaysShowImageSizeBadgeChanged} />
              </FormControl>
              <FormControl>
                <FormLabel>{t('gallery.showArchivedBoards')}</FormLabel>
                <Checkbox isChecked={shouldShowArchivedBoards} onChange={handleChangeShouldShowArchivedBoardsChanged} />
              </FormControl>
            </FormControlGroup>
            <BoardAutoAddSelect />
            <Divider />
            <FormControl w="full">
              <FormLabel flexGrow={1} m={0}>
                {t('gallery.showStarredImagesFirst')}
              </FormLabel>
              <Switch size="sm" isChecked={starredFirst} onChange={onChangeStarredFirst} />
            </FormControl>
            <FormControl>
              <FormLabel flexGrow={1} m={0}>
                {t('gallery.sortDirection')}
              </FormLabel>
              <Combobox
                isSearchable={false}
                value={orderDirValue}
                options={orderDirOptions}
                onChange={onChangeOrderDir}
              />
            </FormControl>
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};

export default memo(GallerySettingsPopover);
