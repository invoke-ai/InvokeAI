import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { useAppSelector } from '../../app/hooks';
import { RootState } from '../../app/store';
import { SDState } from '../sd/sdSlice';
import { validateSeedWeights } from '../sd/util/seedWeightPairs';

const selector = createSelector(
    (state: RootState) => state.sd,
    (sd: SDState) => {
        return {
            prompt: sd.prompt,
            shouldGenerateVariations: sd.shouldGenerateVariations,
            seedWeights: sd.seedWeights,
            maskPath: sd.maskPath,
            initialImagePath: sd.initialImagePath,
            seed: sd.seed,
        };
    },
    {
        memoizeOptions: {
            resultEqualityCheck: isEqual,
        },
    }
);

/*
Returns a function check relevant pieces of state and verify generation will not deterministically fail.

This is used to prevent the 'Generate' button from being clicked.

Other parameter values may cause failure but we rely on input validation for those.
*/
const useCheckParameters = () => {
    const {
        prompt,
        shouldGenerateVariations,
        seedWeights,
        maskPath,
        initialImagePath,
        seed,
    } = useAppSelector(selector);

    const { isProcessing, isConnected } = useAppSelector(
        (state: RootState) => state.system
    );

    // Cannot generate without a prompt
    if (!prompt) {
        return false;
    }

    //  Cannot generate with a mask without img2img
    if (maskPath && !initialImagePath) {
        return false;
    }

    // TODO: job queue
    // Cannot generate if already processing an image
    if (isProcessing) {
        return false;
    }

    // Cannot generate if not connected
    if (!isConnected) {
        return false;
    }

    // Cannot generate variations without valid seed weights
    if (
        shouldGenerateVariations &&
        (!validateSeedWeights(seedWeights) || seed === -1)
    ) {
        return false;
    }

    // All good
    return true;
};

export default useCheckParameters;
