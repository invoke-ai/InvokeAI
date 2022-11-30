import { Flex } from '@chakra-ui/react';

import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/store';

import {
  FacetoolType,
  OptionsState,
  setCodeformerFidelity,
  setFacetoolStrength,
  setFacetoolType,
} from 'features/options/store/optionsSlice';

import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { SystemState } from 'features/system/store/systemSlice';
import IAINumberInput from 'common/components/IAINumberInput';
import IAISelect from 'common/components/IAISelect';
import { FACETOOL_TYPES } from 'app/constants';
import { ChangeEvent } from 'react';

const optionsSelector = createSelector(
  (state: RootState) => state.options,
  (options: OptionsState) => {
    return {
      facetoolStrength: options.facetoolStrength,
      facetoolType: options.facetoolType,
      codeformerFidelity: options.codeformerFidelity,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const systemSelector = createSelector(
  (state: RootState) => state.system,
  (system: SystemState) => {
    return {
      isGFPGANAvailable: system.isGFPGANAvailable,
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
const FaceRestoreOptions = () => {
  const dispatch = useAppDispatch();
  const { facetoolStrength, facetoolType, codeformerFidelity } =
    useAppSelector(optionsSelector);
  const { isGFPGANAvailable } = useAppSelector(systemSelector);

  const handleChangeStrength = (v: number) => dispatch(setFacetoolStrength(v));

  const handleChangeCodeformerFidelity = (v: number) =>
    dispatch(setCodeformerFidelity(v));

  const handleChangeFacetoolType = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setFacetoolType(e.target.value as FacetoolType));

  return (
    <Flex direction={'column'} gap={2}>
      <IAISelect
        label="Type"
        validValues={FACETOOL_TYPES.concat()}
        value={facetoolType}
        onChange={handleChangeFacetoolType}
      />
      <IAINumberInput
        isDisabled={!isGFPGANAvailable}
        label="Strength"
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
          label="Fidelity"
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

export default FaceRestoreOptions;
