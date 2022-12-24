import { createSelector } from '@reduxjs/toolkit';
import { FaRecycle } from 'react-icons/fa';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  OptionsState,
  setShouldLoopback,
} from 'features/options/store/optionsSlice';
import { useTranslation } from 'react-i18next';

const loopbackSelector = createSelector(
  (state: RootState) => state.options,
  (options: OptionsState) => options.shouldLoopback
);

const LoopbackButton = () => {
  const dispatch = useAppDispatch();
  const shouldLoopback = useAppSelector(loopbackSelector);

  const { t } = useTranslation();

  return (
    <IAIIconButton
      aria-label={t('options:toggleLoopback')}
      tooltip={t('options:toggleLoopback')}
      styleClass="loopback-btn"
      asCheckbox={true}
      isChecked={shouldLoopback}
      icon={<FaRecycle />}
      onClick={() => {
        dispatch(setShouldLoopback(!shouldLoopback));
      }}
    />
  );
};

export default LoopbackButton;
