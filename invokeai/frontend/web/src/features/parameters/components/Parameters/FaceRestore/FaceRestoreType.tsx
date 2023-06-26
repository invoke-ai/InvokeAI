import { FACETOOL_TYPES } from 'app/constants';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import {
  FacetoolType,
  setFacetoolType,
} from 'features/parameters/store/postprocessingSlice';
import { useTranslation } from 'react-i18next';

export default function FaceRestoreType() {
  const facetoolType = useAppSelector(
    (state: RootState) => state.postprocessing.facetoolType
  );

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChangeFacetoolType = (v: string) =>
    dispatch(setFacetoolType(v as FacetoolType));

  return (
    <IAIMantineSelect
      label={t('parameters.type')}
      data={FACETOOL_TYPES.concat()}
      value={facetoolType}
      onChange={handleChangeFacetoolType}
    />
  );
}
