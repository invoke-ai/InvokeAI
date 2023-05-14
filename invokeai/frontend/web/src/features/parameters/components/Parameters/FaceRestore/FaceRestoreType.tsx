import { FACETOOL_TYPES } from 'app/constants';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISelect from 'common/components/IAISelect';
import {
  FacetoolType,
  setFacetoolType,
} from 'features/parameters/store/postprocessingSlice';
import { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

export default function FaceRestoreType() {
  const facetoolType = useAppSelector(
    (state: RootState) => state.postprocessing.facetoolType
  );

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChangeFacetoolType = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setFacetoolType(e.target.value as FacetoolType));

  return (
    <IAISelect
      label={t('parameters.type')}
      validValues={FACETOOL_TYPES.concat()}
      value={facetoolType}
      onChange={handleChangeFacetoolType}
    />
  );
}
