import type { FormLabelProps } from '@invoke-ai/ui-library';
import {
  Checkbox,
  CompositeSlider,
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
import {
  autoAssignBoardOnClickChanged,
  setGalleryImageMinimumWidth,
  shouldAutoSwitchChanged,
} from 'features/gallery/store/gallerySlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { RiSettings4Fill } from 'react-icons/ri';

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

  return (
    <Popover isLazy>
      <PopoverTrigger>
        <IconButton
          tooltip={t('gallery.gallerySettings')}
          aria-label={t('gallery.gallerySettings')}
          size="sm"
          icon={<RiSettings4Fill />}
        />
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
                <Switch isChecked={shouldAutoSwitch} onChange={handleChangeAutoSwitch} />
              </FormControl>
              <FormControl>
                <FormLabel>{t('gallery.autoAssignBoardOnClick')}</FormLabel>
                <Checkbox isChecked={autoAssignBoardOnClick} onChange={handleChangeAutoAssignBoardOnClick} />
              </FormControl>
            </FormControlGroup>
            <BoardAutoAddSelect />
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};

export default memo(GallerySettingsPopover);
