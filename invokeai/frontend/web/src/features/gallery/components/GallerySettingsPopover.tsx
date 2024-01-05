import { Flex } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvCheckbox } from 'common/components/InvCheckbox/wrapper';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import {
  InvPopover,
  InvPopoverBody,
  InvPopoverContent,
  InvPopoverTrigger,
} from 'common/components/InvPopover/wrapper';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { InvSwitch } from 'common/components/InvSwitch/wrapper';
import {
  autoAssignBoardOnClickChanged,
  selectGallerySlice,
  setGalleryImageMinimumWidth,
  shouldAutoSwitchChanged,
} from 'features/gallery/store/gallerySlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaWrench } from 'react-icons/fa';

import BoardAutoAddSelect from './Boards/BoardAutoAddSelect';

const selector = createMemoizedSelector(selectGallerySlice, (gallery) => {
  const { galleryImageMinimumWidth, shouldAutoSwitch, autoAssignBoardOnClick } =
    gallery;

  return {
    galleryImageMinimumWidth,
    shouldAutoSwitch,
    autoAssignBoardOnClick,
  };
});

const GallerySettingsPopover = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const { galleryImageMinimumWidth, shouldAutoSwitch, autoAssignBoardOnClick } =
    useAppSelector(selector);

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
    (e: ChangeEvent<HTMLInputElement>) =>
      dispatch(autoAssignBoardOnClickChanged(e.target.checked)),
    [dispatch]
  );

  return (
    <InvPopover isLazy>
      <InvPopoverTrigger>
        <InvIconButton
          tooltip={t('gallery.gallerySettings')}
          aria-label={t('gallery.gallerySettings')}
          size="sm"
          icon={<FaWrench />}
        />
      </InvPopoverTrigger>
      <InvPopoverContent>
        <InvPopoverBody>
          <Flex direction="column" gap={2}>
            <InvControl label={t('gallery.galleryImageSize')}>
              <InvSlider
                value={galleryImageMinimumWidth}
                onChange={handleChangeGalleryImageMinimumWidth}
                min={45}
                max={256}
                defaultValue={90}
              />
            </InvControl>
            <InvControl label={t('gallery.autoSwitchNewImages')}>
              <InvSwitch
                isChecked={shouldAutoSwitch}
                onChange={handleChangeAutoSwitch}
              />
            </InvControl>
            <InvControl label={t('gallery.autoAssignBoardOnClick')}>
              <InvCheckbox
                isChecked={autoAssignBoardOnClick}
                onChange={handleChangeAutoAssignBoardOnClick}
              />
            </InvControl>
            <BoardAutoAddSelect />
          </Flex>
        </InvPopoverBody>
      </InvPopoverContent>
    </InvPopover>
  );
};

export default memo(GallerySettingsPopover);
