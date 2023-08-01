import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIPopover from 'common/components/IAIPopover';
import IAISimpleCheckbox from 'common/components/IAISimpleCheckbox';
import IAISlider from 'common/components/IAISlider';
import {
  autoAssignBoardOnClickChanged,
  setGalleryImageMinimumWidth,
  shouldAutoSwitchChanged,
} from 'features/gallery/store/gallerySlice';
import { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { FaWrench } from 'react-icons/fa';
import BoardAutoAddSelect from './Boards/BoardAutoAddSelect';

const selector = createSelector(
  [stateSelector],
  (state) => {
    const {
      galleryImageMinimumWidth,
      shouldAutoSwitch,
      autoAssignBoardOnClick,
    } = state.gallery;

    return {
      galleryImageMinimumWidth,
      shouldAutoSwitch,
      autoAssignBoardOnClick,
    };
  },
  defaultSelectorOptions
);

const GallerySettingsPopover = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const { galleryImageMinimumWidth, shouldAutoSwitch, autoAssignBoardOnClick } =
    useAppSelector(selector);

  const handleChangeGalleryImageMinimumWidth = (v: number) => {
    dispatch(setGalleryImageMinimumWidth(v));
  };

  return (
    <IAIPopover
      triggerComponent={
        <IAIIconButton
          tooltip={t('gallery.gallerySettings')}
          aria-label={t('gallery.gallerySettings')}
          size="sm"
          icon={<FaWrench />}
        />
      }
    >
      <Flex direction="column" gap={4}>
        <IAISlider
          value={galleryImageMinimumWidth}
          onChange={handleChangeGalleryImageMinimumWidth}
          min={32}
          max={256}
          hideTooltip={true}
          label={t('gallery.galleryImageSize')}
          withReset
          handleReset={() => dispatch(setGalleryImageMinimumWidth(64))}
        />
        <IAISimpleCheckbox
          label={t('gallery.autoSwitchNewImages')}
          isChecked={shouldAutoSwitch}
          onChange={(e: ChangeEvent<HTMLInputElement>) =>
            dispatch(shouldAutoSwitchChanged(e.target.checked))
          }
        />
        <IAISimpleCheckbox
          label={t('gallery.autoAssignBoardOnClick')}
          isChecked={autoAssignBoardOnClick}
          onChange={(e: ChangeEvent<HTMLInputElement>) =>
            dispatch(autoAssignBoardOnClickChanged(e.target.checked))
          }
        />
        <BoardAutoAddSelect />
      </Flex>
    </IAIPopover>
  );
};

export default GallerySettingsPopover;
