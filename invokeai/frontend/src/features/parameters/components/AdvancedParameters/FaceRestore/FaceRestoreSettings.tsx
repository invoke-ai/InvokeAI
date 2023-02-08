import { Flex } from '@chakra-ui/react';

import { useAppDispatch, useAppSelector } from 'app/storeHooks';

import { FacetoolType } from 'features/parameters/store/postprocessingSlice';

import {
  setCodeformerFidelity,
  setFacetoolStrength,
  setFacetoolType,
} from 'features/parameters/store/postprocessingSlice';

import { createSelector } from '@reduxjs/toolkit';
import { FACETOOL_TYPES } from 'app/constants';
import IAINumberInput from 'common/components/IAINumberInput';
import IAISelect from 'common/components/IAISelect';
import { postprocessingSelector } from 'features/parameters/store/postprocessingSelectors';
import { systemSelector } from 'features/system/store/systemSelectors';
import { isEqual } from 'lodash';
import { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

const optionsSelector = createSelector(
  [postprocessingSelector, systemSelector],
  (
    { facetoolStrength, facetoolType, codeformerFidelity },
    { isGFPGANAvailable }
  ) => {
    return {
      facetoolStrength,
      facetoolType,
      codeformerFidelity,
      isGFPGANAvailable,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

/**
 * Displays face-fixing/GFPGAN options (strength).
 */
const FaceRestoreSettings = () => {
  const dispatch = useAppDispatch();
  const {
    facetoolStrength,
    facetoolType,
    codeformerFidelity,
    isGFPGANAvailable,
  } = useAppSelector(optionsSelector);

  const handleChangeStrength = (v: number) => dispatch(setFacetoolStrength(v));

  const handleChangeCodeformerFidelity = (v: number) =>
    dispatch(setCodeformerFidelity(v));

  const handleChangeFacetoolType = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setFacetoolType(e.target.value as FacetoolType));

  const { t } = useTranslation();

  return (
    <Flex direction={'column'} gap={2}>
      <IAISelect
        label={t('parameters:type')}
        validValues={FACETOOL_TYPES.concat()}
        value={facetoolType}
        onChange={handleChangeFacetoolType}
      />
      <IAINumberInput
        isDisabled={!isGFPGANAvailable}
        label={t('parameters:strength')}
        step={0.05}
        min={0}
        max={1}
        onChange={handleChangeStrength}
        value={facetoolStrength}
        width="90px"
        isInteger={false}
      />
      {facetoolType === 'codeformer' && (
        <IAINumberInput
          isDisabled={!isGFPGANAvailable}
          label={t('parameters:codeformerFidelity')}
          step={0.05}
          min={0}
          max={1}
          onChange={handleChangeCodeformerFidelity}
          value={codeformerFidelity}
          width="90px"
          isInteger={false}
        />
      )}
    </Flex>
  );
};

export default FaceRestoreSettings;
