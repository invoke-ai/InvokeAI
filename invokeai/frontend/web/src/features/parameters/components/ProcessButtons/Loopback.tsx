import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { postprocessingSelector } from 'features/parameters/store/postprocessingSelectors';
import { setShouldLoopback } from 'features/parameters/store/postprocessingSlice';
import { useTranslation } from 'react-i18next';
import { FaRecycle } from 'react-icons/fa';

const loopbackSelector = createSelector(
  postprocessingSelector,
  ({ shouldLoopback }) => shouldLoopback
);

const LoopbackButton = () => {
  const dispatch = useAppDispatch();
  const shouldLoopback = useAppSelector(loopbackSelector);

  const { t } = useTranslation();

  return (
    <IAIIconButton
      aria-label={t('parameters.toggleLoopback')}
      tooltip={t('parameters.toggleLoopback')}
      isChecked={shouldLoopback}
      icon={<FaRecycle />}
      onClick={() => {
        dispatch(setShouldLoopback(!shouldLoopback));
      }}
    />
  );
};

export default LoopbackButton;
