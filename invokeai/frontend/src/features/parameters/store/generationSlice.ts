import { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import * as InvokeAI from 'app/invokeai';
import { getPromptAndNegative } from 'common/util/getPromptAndNegative';
import promptToString from 'common/util/promptToString';
import { seedWeightsToString } from 'common/util/seedWeightPairs';
import { clamp } from 'lodash';

export interface GenerationState {
  cfgScale: number;
  height: number;
  img2imgStrength: number;
  infillMethod: string;
  initialImage?: InvokeAI.Image | string; // can be an Image or url
  iterations: number;
  maskPath: string;
  perlin: number;
  prompt: string;
  negativePrompt: string;
  lorasInUse: string[];
  localTextualInversionTriggers: string[];
  huggingFaceTextualInversionConcepts: string[];
  textualInversionsInUse: string[];
  sampler: string;
  seamBlur: number;
  seamless: boolean;
  seamSize: number;
  seamSteps: number;
  seamStrength: number;
  seed: number;
  seedWeights: string;
  shouldFitToWidthHeight: boolean;
  shouldGenerateVariations: boolean;
  shouldRandomizeSeed: boolean;
  steps: number;
  threshold: number;
  tileSize: number;
  variationAmount: number;
  width: number;
  shouldUseSymmetry: boolean;
  horizontalSymmetrySteps: number;
  verticalSymmetrySteps: number;
}

const initialGenerationState: GenerationState = {
  cfgScale: 7.5,
  height: 512,
  img2imgStrength: 0.75,
  infillMethod: 'patchmatch',
  iterations: 1,
  maskPath: '',
  perlin: 0,
  prompt: '',
  negativePrompt: '',
  lorasInUse: [],
  localTextualInversionTriggers: [],
  huggingFaceTextualInversionConcepts: [],
  textualInversionsInUse: [],
  sampler: 'k_lms',
  seamBlur: 16,
  seamless: false,
  seamSize: 96,
  seamSteps: 30,
  seamStrength: 0.7,
  seed: 0,
  seedWeights: '',
  shouldFitToWidthHeight: true,
  shouldGenerateVariations: false,
  shouldRandomizeSeed: true,
  steps: 50,
  threshold: 0,
  tileSize: 32,
  variationAmount: 0.1,
  width: 512,
  shouldUseSymmetry: false,
  horizontalSymmetrySteps: 0,
  verticalSymmetrySteps: 0,
};

const initialState: GenerationState = initialGenerationState;

const loraExists = (state: GenerationState, lora: string) => {
  const loraRegex = new RegExp(`withLora\\(${lora},?\\s*([^\\)]+)?\\)`);
  if (state.prompt.match(loraRegex)) return true;
  return false;
};

const textualInversionExists = (
  state: GenerationState,
  textualInversion: string
) => {
  const textualInversionRegex = new RegExp(textualInversion);
  if (state.prompt.match(textualInversionRegex)) return true;
  return false;
};

export const generationSlice = createSlice({
  name: 'generation',
  initialState,
  reducers: {
    setPrompt: (state, action: PayloadAction<string | InvokeAI.Prompt>) => {
      const newPrompt = action.payload;

      if (typeof newPrompt === 'string') {
        state.prompt = newPrompt;
      } else {
        state.prompt = promptToString(newPrompt);
      }
    },
    setNegativePrompt: (
      state,
      action: PayloadAction<string | InvokeAI.Prompt>
    ) => {
      const newPrompt = action.payload;
      if (typeof newPrompt === 'string') {
        state.negativePrompt = newPrompt;
      } else {
        state.negativePrompt = promptToString(newPrompt);
      }
    },
    setLorasInUse: (state, action: PayloadAction<string>) => {
      const newLora = action.payload;
      const loras = [...state.lorasInUse];

      if (loraExists(state, newLora)) {
        const loraRegex = new RegExp(
          `withLora\\(${newLora},?\\s*([^\\)]+)?\\)`,
          'g'
        );
        const newPrompt = state.prompt.replaceAll(loraRegex, '');
        state.prompt = newPrompt;

        if (loras.includes(newLora)) {
          const newLoraIndex = loras.indexOf(newLora);
          if (newLoraIndex > -1) loras.splice(newLoraIndex, 1);
        }
      } else {
        state.prompt = `${state.prompt} withLora(${newLora},0.75)`;
        if (!loras.includes(newLora)) loras.push(newLora);
      }
      state.lorasInUse = loras;
    },
    handlePromptCheckers: (
      state,
      action: PayloadAction<string | InvokeAI.Prompt>
    ) => {
      const newPrompt = action.payload;

      // Tackle User Typed Lora Syntax
      let lorasInUse = [...state.lorasInUse]; // Get Loras In Prompt
      const loraRegex = /withLora\(([^\\)]+)\)/g; // Scan For Lora Syntax
      if (typeof newPrompt === 'string') {
        const loraMatches = [...newPrompt.matchAll(loraRegex)]; // Match All Lora Syntaxes
        if (loraMatches.length > 0) {
          lorasInUse = []; // Reset Loras In Use
          loraMatches.forEach((loraMatch) => {
            const loraName = loraMatch[1].split(',')[0];
            if (!lorasInUse.includes(loraName)) lorasInUse.push(loraName); // Add Loras In Prompt
          });
        } else {
          lorasInUse = []; // If No Matches, Remove Loras In Use
        }
      }
      state.lorasInUse = lorasInUse;

      // Tackle User Typed Textual Inversion
      let textualInversionsInUse = [...state.textualInversionsInUse]; // Get Words In Prompt
      const textualInversionRegex = /([\w<>!@%&*_-]+)/g; // Scan For Each Word
      if (typeof newPrompt === 'string') {
        const textualInversionMatches = [
          ...newPrompt.matchAll(textualInversionRegex),
        ]; // Match All Words
        if (textualInversionMatches.length > 0) {
          textualInversionsInUse = []; // Reset Textual Inversions In Use
          console.log(textualInversionMatches);
          textualInversionMatches.forEach((textualInversionMatch) => {
            const textualInversionName = textualInversionMatch[0];
            console.log(textualInversionName);
            if (
              !textualInversionsInUse.includes(textualInversionName) &&
              (state.localTextualInversionTriggers.includes(
                textualInversionName
              ) ||
                state.huggingFaceTextualInversionConcepts.includes(
                  textualInversionName
                ))
            )
              textualInversionsInUse.push(textualInversionName); // Add Textual Inversions In Prompt
          });
        } else {
          textualInversionsInUse = []; // If No Matches, Remove Textual Inversions In Use
        }
      }

      console.log([...state.huggingFaceTextualInversionConcepts]);
      state.textualInversionsInUse = textualInversionsInUse;
    },
    setTextualInversionsInUse: (state, action: PayloadAction<string>) => {
      const newTextualInversion = action.payload;
      const textualInversions = [...state.textualInversionsInUse];

      if (textualInversionExists(state, newTextualInversion)) {
        const textualInversionRegex = new RegExp(newTextualInversion, 'g');
        const newPrompt = state.prompt.replaceAll(textualInversionRegex, '');
        state.prompt = newPrompt;

        if (textualInversions.includes(newTextualInversion)) {
          const newTIIndex = textualInversions.indexOf(newTextualInversion);
          if (newTIIndex > -1) textualInversions.splice(newTIIndex, 1);
        }
      } else {
        state.prompt = `${state.prompt} ${newTextualInversion}`;
        if (!textualInversions.includes(newTextualInversion))
          textualInversions.push(newTextualInversion);
      }
      state.lorasInUse = textualInversions;

      state.textualInversionsInUse = textualInversions;
    },
    setLocalTextualInversionTriggers: (
      state,
      action: PayloadAction<string[]>
    ) => {
      state.localTextualInversionTriggers = action.payload;
    },
    setHuggingFaceTextualInversionConcepts: (
      state,
      action: PayloadAction<string[]>
    ) => {
      state.huggingFaceTextualInversionConcepts = action.payload;
    },
    setIterations: (state, action: PayloadAction<number>) => {
      state.iterations = action.payload;
    },
    setSteps: (state, action: PayloadAction<number>) => {
      state.steps = action.payload;
    },
    clampSymmetrySteps: (state) => {
      state.horizontalSymmetrySteps = clamp(
        state.horizontalSymmetrySteps,
        0,
        state.steps
      );
      state.verticalSymmetrySteps = clamp(
        state.verticalSymmetrySteps,
        0,
        state.steps
      );
    },
    setCfgScale: (state, action: PayloadAction<number>) => {
      state.cfgScale = action.payload;
    },
    setThreshold: (state, action: PayloadAction<number>) => {
      state.threshold = action.payload;
    },
    setPerlin: (state, action: PayloadAction<number>) => {
      state.perlin = action.payload;
    },
    setHeight: (state, action: PayloadAction<number>) => {
      state.height = action.payload;
    },
    setWidth: (state, action: PayloadAction<number>) => {
      state.width = action.payload;
    },
    setSampler: (state, action: PayloadAction<string>) => {
      state.sampler = action.payload;
    },
    setSeed: (state, action: PayloadAction<number>) => {
      state.seed = action.payload;
      state.shouldRandomizeSeed = false;
    },
    setImg2imgStrength: (state, action: PayloadAction<number>) => {
      state.img2imgStrength = action.payload;
    },
    setMaskPath: (state, action: PayloadAction<string>) => {
      state.maskPath = action.payload;
    },
    setSeamless: (state, action: PayloadAction<boolean>) => {
      state.seamless = action.payload;
    },
    setShouldFitToWidthHeight: (state, action: PayloadAction<boolean>) => {
      state.shouldFitToWidthHeight = action.payload;
    },
    resetSeed: (state) => {
      state.seed = -1;
    },
    setParameter: (
      state,
      action: PayloadAction<{ key: string; value: string | number | boolean }>
    ) => {
      // TODO: This probably needs to be refactored.
      // TODO: This probably also needs to be fixed after the reorg.
      const { key, value } = action.payload;
      const temp = { ...state, [key]: value };
      if (key === 'seed') {
        temp.shouldRandomizeSeed = false;
      }
      return temp;
    },
    setShouldGenerateVariations: (state, action: PayloadAction<boolean>) => {
      state.shouldGenerateVariations = action.payload;
    },
    setVariationAmount: (state, action: PayloadAction<number>) => {
      state.variationAmount = action.payload;
    },
    setSeedWeights: (state, action: PayloadAction<string>) => {
      state.seedWeights = action.payload;
      state.shouldGenerateVariations = true;
      state.variationAmount = 0;
    },
    setAllTextToImageParameters: (
      state,
      action: PayloadAction<InvokeAI.Metadata>
    ) => {
      const {
        sampler,
        prompt,
        seed,
        variations,
        steps,
        cfg_scale,
        threshold,
        perlin,
        seamless,
        hires_fix,
        width,
        height,
      } = action.payload.image;

      if (variations && variations.length > 0) {
        state.seedWeights = seedWeightsToString(variations);
        state.shouldGenerateVariations = true;
        state.variationAmount = 0;
      } else {
        state.shouldGenerateVariations = false;
      }

      if (seed) {
        state.seed = seed;
        state.shouldRandomizeSeed = false;
      }

      if (prompt) state.prompt = promptToString(prompt);
      if (sampler) state.sampler = sampler;
      if (steps) state.steps = steps;
      if (cfg_scale) state.cfgScale = cfg_scale;
      if (typeof threshold === 'undefined') {
        state.threshold = 0;
      } else {
        state.threshold = threshold;
      }
      if (typeof perlin === 'undefined') {
        state.perlin = 0;
      } else {
        state.perlin = perlin;
      }
      if (typeof seamless === 'boolean') state.seamless = seamless;
      // if (typeof hires_fix === 'boolean') state.hiresFix = hires_fix; // TODO: Needs to be fixed after reorg
      if (width) state.width = width;
      if (height) state.height = height;
    },
    setAllImageToImageParameters: (
      state,
      action: PayloadAction<InvokeAI.Metadata>
    ) => {
      const { type, strength, fit, init_image_path, mask_image_path } =
        action.payload.image;

      if (type === 'img2img') {
        if (init_image_path) state.initialImage = init_image_path;
        if (mask_image_path) state.maskPath = mask_image_path;
        if (strength) state.img2imgStrength = strength;
        if (typeof fit === 'boolean') state.shouldFitToWidthHeight = fit;
      }
    },
    setAllParameters: (state, action: PayloadAction<InvokeAI.Metadata>) => {
      const {
        type,
        sampler,
        prompt,
        seed,
        variations,
        steps,
        cfg_scale,
        threshold,
        perlin,
        seamless,
        hires_fix,
        width,
        height,
        strength,
        fit,
        init_image_path,
        mask_image_path,
      } = action.payload.image;

      if (type === 'img2img') {
        if (init_image_path) state.initialImage = init_image_path;
        if (mask_image_path) state.maskPath = mask_image_path;
        if (strength) state.img2imgStrength = strength;
        if (typeof fit === 'boolean') state.shouldFitToWidthHeight = fit;
      }

      if (variations && variations.length > 0) {
        state.seedWeights = seedWeightsToString(variations);
        state.shouldGenerateVariations = true;
        state.variationAmount = 0;
      } else {
        state.shouldGenerateVariations = false;
      }

      if (seed) {
        state.seed = seed;
        state.shouldRandomizeSeed = false;
      }

      if (prompt) {
        const [promptOnly, negativePrompt] = getPromptAndNegative(prompt);
        if (promptOnly) state.prompt = promptOnly;
        negativePrompt
          ? (state.negativePrompt = negativePrompt)
          : (state.negativePrompt = '');
      }

      if (sampler) state.sampler = sampler;
      if (steps) state.steps = steps;
      if (cfg_scale) state.cfgScale = cfg_scale;
      if (typeof threshold === 'undefined') {
        state.threshold = 0;
      } else {
        state.threshold = threshold;
      }
      if (typeof perlin === 'undefined') {
        state.perlin = 0;
      } else {
        state.perlin = perlin;
      }
      if (typeof seamless === 'boolean') state.seamless = seamless;
      if (width) state.width = width;
      if (height) state.height = height;

      // state.shouldRunESRGAN = false; // TODO: Needs to be fixed after reorg
      // state.shouldRunFacetool = false; // TODO: Needs to be fixed after reorg
    },
    resetParametersState: (state) => {
      return {
        ...state,
        ...initialGenerationState,
      };
    },
    setShouldRandomizeSeed: (state, action: PayloadAction<boolean>) => {
      state.shouldRandomizeSeed = action.payload;
    },
    setInitialImage: (
      state,
      action: PayloadAction<InvokeAI.Image | string>
    ) => {
      state.initialImage = action.payload;
    },
    clearInitialImage: (state) => {
      state.initialImage = undefined;
    },
    setSeamSize: (state, action: PayloadAction<number>) => {
      state.seamSize = action.payload;
    },
    setSeamBlur: (state, action: PayloadAction<number>) => {
      state.seamBlur = action.payload;
    },
    setSeamStrength: (state, action: PayloadAction<number>) => {
      state.seamStrength = action.payload;
    },
    setSeamSteps: (state, action: PayloadAction<number>) => {
      state.seamSteps = action.payload;
    },
    setTileSize: (state, action: PayloadAction<number>) => {
      state.tileSize = action.payload;
    },
    setInfillMethod: (state, action: PayloadAction<string>) => {
      state.infillMethod = action.payload;
    },
    setShouldUseSymmetry: (state, action: PayloadAction<boolean>) => {
      state.shouldUseSymmetry = action.payload;
    },
    setHorizontalSymmetrySteps: (state, action: PayloadAction<number>) => {
      state.horizontalSymmetrySteps = action.payload;
    },
    setVerticalSymmetrySteps: (state, action: PayloadAction<number>) => {
      state.verticalSymmetrySteps = action.payload;
    },
  },
});

export const {
  clampSymmetrySteps,
  clearInitialImage,
  resetParametersState,
  resetSeed,
  setAllImageToImageParameters,
  setAllParameters,
  setAllTextToImageParameters,
  setCfgScale,
  setHeight,
  setImg2imgStrength,
  setInfillMethod,
  setInitialImage,
  setIterations,
  setMaskPath,
  setParameter,
  setPerlin,
  setPrompt,
  setNegativePrompt,
  setLorasInUse,
  setLocalTextualInversionTriggers,
  setHuggingFaceTextualInversionConcepts,
  setTextualInversionsInUse,
  handlePromptCheckers,
  setSampler,
  setSeamBlur,
  setSeamless,
  setSeamSize,
  setSeamSteps,
  setSeamStrength,
  setSeed,
  setSeedWeights,
  setShouldFitToWidthHeight,
  setShouldGenerateVariations,
  setShouldRandomizeSeed,
  setSteps,
  setThreshold,
  setTileSize,
  setVariationAmount,
  setWidth,
  setShouldUseSymmetry,
  setHorizontalSymmetrySteps,
  setVerticalSymmetrySteps,
} = generationSlice.actions;

export default generationSlice.reducer;
